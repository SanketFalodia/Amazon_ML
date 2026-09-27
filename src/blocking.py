"""Recall-first candidate generation via a UNION of cheap, complementary blocking keys.

Two-pass construction (fixes the OOM you hit at "[block] building train
candidates (2206821 x 10320219)"):

  Pass 1 counts each key's GLOBAL document frequency across the whole
  "others" corpus (S2+S3) using plain int counters -- it NEVER stores an
  id list, only a running count per key.

  Pass 2 builds the actual inverted index, but only inserts a row's id
  under a key if that key's global df is within [df_min, df_max] for its
  type. Keys that are too rare (a typo-only token, df < rare_token_df_min)
  or too common (df > *_df_max -- a generic 3-gram/word that shows up in a
  huge fraction of a 10M+ row corpus) are never turned into an id list.

  Why this was the crash: the previous version applied the df cap only at
  LOOKUP time. Every key -- including "mega" keys like common 3-grams
  ("ind", "com", "pvt"...) that appear in a large fraction of a 10M+ row
  table -- was still fully materialized as a list of entity ids during the
  SCAN pass, then thrown away later. A handful of such keys, each holding
  millions of ids, is enough on its own to exhaust Kaggle's RAM before the
  cap is ever consulted. Moving the cap to build time means no list this
  module ever holds can exceed its configured df_max -- memory is bounded
  by (number of surviving keys) x (max cap), not by corpus size.

  Everything else (chunked itertuples() streaming, fork-based Pool sharing
  the source DataFrame via copy-on-write, per-key-type caps) is unchanged
  from the previous version.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from multiprocessing import get_context
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
    return 0, None  # pin:/zip: stay uncapped -- exact numeric match, rarely a mega-block


_G_DF = None
_G_CFG = None
_G_KEEP = None
_G_INV = None


def _pool_init_count(df, cfg):
    global _G_DF, _G_CFG
    _G_DF, _G_CFG = df, cfg


def _count_range(rng: tuple[int, int]) -> dict[str, int]:
    start, end = rng
    df, cfg = _G_DF, _G_CFG
    counts: Counter = Counter()
    for row in df.iloc[start:end].itertuples(index=False):
        counts.update(_keys_row(row, cfg))
    return counts


def _pool_init_scan(df, cfg, keep):
    global _G_DF, _G_CFG, _G_KEEP
    _G_DF, _G_CFG, _G_KEEP = df, cfg, keep


def _scan_range(rng: tuple[int, int]) -> dict[str, list[str]]:
    start, end = rng
    df, cfg, keep = _G_DF, _G_CFG, _G_KEEP
    inv: dict[str, list[str]] = defaultdict(list)
    for row in df.iloc[start:end].itertuples(index=False):
        for k in _keys_row(row, cfg):
            if k in keep:
                inv[k].append(row.entity_id)
    return inv


def _pool_init_lookup(inv, cfg, df):
    global _G_INV, _G_CFG, _G_DF
    _G_INV, _G_CFG, _G_DF = inv, cfg, df


def _lookup_range(rng: tuple[int, int]) -> dict[str, list[str]]:
    start, end = rng
    inv, cfg, df = _G_INV, _G_CFG, _G_DF
    out: dict[str, list[str]] = {}
    for row in df.iloc[start:end].itertuples(index=False):
        see: set[str] = set()
        for k in _keys_row(row, cfg):
            ids = inv.get(k)
            if ids:
                see.update(ids)
        out[row.entity_id] = sorted(see)
    return out


def _build_df_counts(others: pd.DataFrame, cfg: dict, n_jobs: int, chunk_size: int, ctx) -> Counter:
    n_other = len(others)
    counts: Counter = Counter()
    if n_jobs > 1 and n_other > chunk_size:
        ranges = list(_chunks_idx(n_other, chunk_size))
        with ctx.Pool(n_jobs, initializer=_pool_init_count, initargs=(others, cfg)) as pool:
            for part in pool.imap_unordered(_count_range, ranges):
                counts.update(part)
                del part
    else:
        for row in others.itertuples(index=False):
            counts.update(_keys_row(row, cfg))
    return counts


def build_candidates(s1n: pd.DataFrame, others: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """others = concatenated+normalized S2 & S3. Returns s1_id -> [ids]."""
    n_jobs = cfg.get("n_jobs") or os.cpu_count() or 1
    chunk_size = cfg.get("chunk_size", 200_000)
    ctx = _ctx()

    # --- Pass 1: global df per key -- counts only, no id lists -> small memory. ---
    df_counts = _build_df_counts(others, cfg, n_jobs, chunk_size, ctx)
    keep = {
        k for k, c in df_counts.items()
        if c >= _bounds_for(k, cfg)[0]
        and (_bounds_for(k, cfg)[1] is None or c <= _bounds_for(k, cfg)[1])
    }
    print(f"[block]   {len(df_counts)} distinct keys seen, keeping {len(keep)} "
          f"after df_min/df_max (dropped {len(df_counts) - len(keep)})")
    del df_counts

    # --- Pass 2: build inverted index, but ONLY for keys that survived the cap. ---
    # No list built here can ever exceed its configured df_max.
    n_other = len(others)
    inv: dict[str, list[str]] = defaultdict(list)
    if n_jobs > 1 and n_other > chunk_size:
        ranges = list(_chunks_idx(n_other, chunk_size))
        with ctx.Pool(n_jobs, initializer=_pool_init_scan, initargs=(others, cfg, keep)) as pool:
            for part in pool.imap_unordered(_scan_range, ranges):
                for k, v in part.items():
                    inv[k].extend(v)
                del part
    else:
        for row in others.itertuples(index=False):
            for k in _keys_row(row, cfg):
                if k in keep:
                    inv[k].append(row.entity_id)
    del keep

    # --- Lookup: union candidates across every key the S1 row generates. ---
    n_s1 = len(s1n)
    out: dict[str, list[str]] = {}
    if n_jobs > 1 and n_s1 > chunk_size:
        ranges = list(_chunks_idx(n_s1, chunk_size))
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
            out[row.entity_id] = sorted(see)

    return out