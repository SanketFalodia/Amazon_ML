"""Recall-first candidate generation via a UNION of cheap, complementary blocking keys.

Two-pass construction (fixes the OOM at "[block] building train
candidates (2206821 x 10320219)"):

  Pass 1 counts each key's GLOBAL document frequency across the whole
  "others" corpus (S2+S3) using plain int counters -- it NEVER stores an
  id list, only a running count per key.

  Pass 2 builds the actual inverted index, but only inserts a row's id
  under a key if that key's global df is within [df_min, df_max] for its
  type. Keys that are too rare or too common are never turned into an id list.

`others` is accepted as a DataFrame OR a list/tuple of DataFrames (e.g.
[s2n, s3n]) and is NEVER internally concatenated -- pd.concat() on two
multi-million-row frames briefly holds both source frames AND the combined
result simultaneously, which is enough on its own to exhaust RAM at this
scale. Accepting a list and iterating each frame's own chunks sidesteps
that spike entirely.

THIRD FIX (this version): pin:/zip: keys were the one channel with NO
df_max, on the theory that a postal code is "rarely a mega-block". At
11M+ records that assumption breaks -- a single PIN/ZIP shared by a dense
area (very common in India, where a 6-digit PIN can cover a whole
district) becomes an unbounded bucket. In the lookup phase, EVERY S1
entity sharing that key unions the whole bucket into its own candidate
set, so a 300k-id bucket shared by 100k entities alone produces ~3*10^10
pointer-slots -- this is almost certainly what actually exhausted RAM
right after the "distinct keys ... keeping ..." checkpoint, since that
checkpoint prints fine and the crash happens silently afterward, in the
lookup phase, with no further output.

Two changes fix this:
  1. `_bounds_for` now caps pin:/zip: the same way every other channel is
     capped (new `pin_token_df_max` config key, defaults to 300).
  2. A hard `max_candidates_per_entity` ceiling is applied in the lookup
     phase, as a backstop against ANY future mega-bucket, known or not --
     it truncates (not silently grows) any one entity's final candidate
     list. Off by default (unset in config = unbounded, unchanged
     behavior) so this is safe to drop in even before you've set it.

A diagnostic line was also added after Pass 1: it prints the 5 largest
surviving buckets by count, so you can directly confirm on your own data
whether a pin:/zip: (or any other) key was in fact the mega-bucket.

Everything else (chunked itertuples() streaming, fork-based Pool sharing
each source DataFrame via copy-on-write, the two-pass count/cap logic) is
unchanged.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from multiprocessing import get_context
import heapq
import os
import pandas as pd


def _ctx():
    try:
        return get_context("fork")           # cheap sharing; default on Linux/Kaggle
    except ValueError:
        return get_context()                  # fallback (e.g. Windows -> spawn)


def _chunks_idx(n: int, size: int):
    for i in range(0, n, size):
        yield (i, min(i + size, n))


def _multi_ranges(dfs: list[pd.DataFrame], chunk_size: int) -> list[tuple[int, int, int]]:
    """(df_index, start, end) ranges spanning every frame in dfs, so a single
    Pool.imap_unordered call can chunk-process all of them without ever
    concatenating them first."""
    ranges = []
    for i, df in enumerate(dfs):
        for start, end in _chunks_idx(len(df), chunk_size):
            ranges.append((i, start, end))
    return ranges


def _keys_row(row, cfg: dict) -> set[str]:
    """row: a namedtuple from itertuples(index=False) -- attribute access only."""
    keys: set[str] = set()
    for t in row.name_no_legal.split():
        keys.add(f"ntok:{t}")
    srt = row.name_sorted.split()
    if srt:
        keys.add(f"nfirst:{srt[0]}")
        if len(srt) > 1:
            keys.add(f"nfirst2:{srt[0]}_{srt[1]}")
    if row.name_meta:
        keys.add(f"meta:{row.name_meta}")
    if len(row.name_init) >= 2:
        keys.add(f"init:{row.name_init}")
    if row.pincode:
        keys.add(f"pin:{row.pincode}")
    if row.zip5:
        keys.add(f"zip:{row.zip5}")
    if row.city and row.state:
        keys.add(f"cs:{row.city}_{row.state}")
    for t in set(row.addr_no_digits.split()):
        keys.add(f"atok:{t}")
    comp = row.name_compact
    n = cfg["char_ngram_n"]
    if len(comp) >= n:
        for i in range(len(comp) - n + 1):
            keys.add(f"ng:{comp[i:i + n]}")
    return keys


def _bounds_for(k: str, cfg: dict) -> tuple[int, int | None]:
    """(df_min, df_max) for a key type. df_max=None means uncapped."""
    if k.startswith(("ntok:", "ng:")):
        return cfg.get("rare_token_df_min", 0), cfg["rare_token_df_max"]
    if k.startswith("atok:"):
        return 0, cfg["addr_token_df_max"]
    if k.startswith(("cs:", "nfirst:", "nfirst2:", "meta:", "init:")):
        return 0, cfg.get("loc_token_df_max", cfg["rare_token_df_max"])
    if k.startswith(("pin:", "zip:")):
        # FIX: was `return 0, None` (uncapped). See module docstring -- a
        # PIN/ZIP shared by thousands of entities was an unbounded mega-
        # bucket and is the most likely actual cause of the RAM blowup.
        return 0, cfg.get("pin_token_df_max", 300)
    return 0, None


_G_DFS = None   # list/tuple of DataFrames, indexed by df_idx -- never concatenated
_G_CFG = None
_G_KEEP = None
_G_INV = None


def _pool_init_count(dfs, cfg):
    global _G_DFS, _G_CFG
    _G_DFS, _G_CFG = dfs, cfg


def _count_range(rng: tuple[int, int, int]) -> dict[str, int]:
    df_idx, start, end = rng
    dfs, cfg = _G_DFS, _G_CFG
    counts: Counter = Counter()
    for row in dfs[df_idx].iloc[start:end].itertuples(index=False):
        counts.update(_keys_row(row, cfg))
    return counts


def _pool_init_scan(dfs, cfg, keep):
    global _G_DFS, _G_CFG, _G_KEEP
    _G_DFS, _G_CFG, _G_KEEP = dfs, cfg, keep


def _scan_range(rng: tuple[int, int, int]) -> dict[str, list[str]]:
    df_idx, start, end = rng
    dfs, cfg, keep = _G_DFS, _G_CFG, _G_KEEP
    inv: dict[str, list[str]] = defaultdict(list)
    for row in dfs[df_idx].iloc[start:end].itertuples(index=False):
        for k in _keys_row(row, cfg):
            if k in keep:
                inv[k].append(row.entity_id)
    return inv


def _pool_init_lookup(inv, cfg, df):
    global _G_INV, _G_CFG, _G_DFS
    _G_INV, _G_CFG, _G_DFS = inv, cfg, (df,)


def _lookup_range(rng: tuple[int, int, int]) -> dict[str, list[str]]:
    _, start, end = rng
    inv, cfg, df = _G_INV, _G_CFG, _G_DFS[0]
    cap = cfg.get("max_candidates_per_entity")  # NEW -- None/0 = unbounded (old behavior)
    out: dict[str, list[str]] = {}
    for row in df.iloc[start:end].itertuples(index=False):
        see: set[str] = set()
        for k in _keys_row(row, cfg):
            ids = inv.get(k)
            if ids:
                see.update(ids)
        result = sorted(see)
        if cap:
            result = result[:cap]           # NEW -- hard per-entity ceiling
        out[row.entity_id] = result
    return out


def _build_df_counts(dfs: list[pd.DataFrame], cfg: dict, n_jobs: int,
                      chunk_size: int, ctx) -> Counter:
    counts: Counter = Counter()
    total = sum(len(df) for df in dfs)
    if n_jobs > 1 and total > chunk_size:
        ranges = _multi_ranges(dfs, chunk_size)
        with ctx.Pool(n_jobs, initializer=_pool_init_count, initargs=(dfs, cfg)) as pool:
            for part in pool.imap_unordered(_count_range, ranges):
                counts.update(part)
                del part
    else:
        for df in dfs:
            for row in df.itertuples(index=False):
                counts.update(_keys_row(row, cfg))
    return counts


def build_candidates(s1n: pd.DataFrame, others, cfg: dict) -> dict[str, list[str]]:
    """others: a DataFrame OR a list/tuple of DataFrames (e.g. [s2n, s3n]).
    NEVER internally concatenated. Returns s1_id -> [ids]."""
    dfs = list(others) if isinstance(others, (list, tuple)) else [others]

    n_jobs = cfg.get("n_jobs") or os.cpu_count() or 1
    chunk_size = cfg.get("chunk_size", 200_000)
    ctx = _ctx()

    # --- Pass 1: global df per key -- counts only, no id lists -> small memory. ---
    df_counts = _build_df_counts(dfs, cfg, n_jobs, chunk_size, ctx)
    keep = {
        k for k, c in df_counts.items()
        if c >= _bounds_for(k, cfg)[0]
        and (_bounds_for(k, cfg)[1] is None or c <= _bounds_for(k, cfg)[1])
    }
    n_other = sum(len(df) for df in dfs)
    print(f"[block]   {len(df_counts)} distinct keys seen across {n_other} rows, "
          f"keeping {len(keep)} after df_min/df_max (dropped {len(df_counts) - len(keep)})")

    # NEW -- diagnostic: shows you directly whether a pin:/zip: (or any other)
    # key was the mega-bucket. Cheap: a single pass with a size-5 heap.
    top = heapq.nlargest(5, ((c, k) for k, c in df_counts.items() if k in keep))
    print("[block]   largest surviving buckets: " +
          ", ".join(f"{k}={c}" for c, k in top))

    del df_counts

    # --- Pass 2: build inverted index, but ONLY for keys that survived the cap. ---
    inv: dict[str, list[str]] = defaultdict(list)
    if n_jobs > 1 and n_other > chunk_size:
        ranges = _multi_ranges(dfs, chunk_size)
        with ctx.Pool(n_jobs, initializer=_pool_init_scan, initargs=(dfs, cfg, keep)) as pool:
            for part in pool.imap_unordered(_scan_range, ranges):
                for k, v in part.items():
                    inv[k].extend(v)
                del part
    else:
        for df in dfs:
            for row in df.itertuples(index=False):
                for k in _keys_row(row, cfg):
                    if k in keep:
                        inv[k].append(row.entity_id)
    del keep

    # --- Lookup: union candidates across every key the S1 row generates. ---
    n_s1 = len(s1n)
    cap = cfg.get("max_candidates_per_entity")  # NEW
    out: dict[str, list[str]] = {}
    if n_jobs > 1 and n_s1 > chunk_size:
        ranges = [(0, s, e) for s, e in _chunks_idx(n_s1, chunk_size)]
        with ctx.Pool(n_jobs, initializer=_pool_init_lookup, initargs=(inv, cfg, s1n)) as pool:
            for part in pool.imap_unordered(_lookup_range, ranges):
                out.update(part)
                del part
    else:
        for row in s1n.itertuples(index=False):
            see: set[str] = set()
            for k in _keys_row(row, cfg):
                ids = inv.get(k)
                if ids:
                    see.update(ids)
            result = sorted(see)
            if cap:
                result = result[:cap]        # NEW
            out[row.entity_id] = result

    return out