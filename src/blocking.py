"""Recall-first candidate generation via a UNION of cheap, complementary blocking keys.

Optimized vs original, in order of impact:
  1. Rows are streamed via itertuples() in bounded chunks (default 200k rows), never
     materialized as a full list of dicts/records for the whole corpus. On a 10M+ row
     "others" (S2+S3) frame, to_dict("records") would build ~10M standalone Python
     dict objects simultaneously -- often MORE memory than the source data itself.
     itertuples() over a small .iloc slice keeps only one chunk's rows materialized
     at a time.
  2. Every key type is capped (df_max), not just ntok:/ng:/atok:. Previously cs:
     (city+state), nfirst:, nfirst2:, meta:, init: were UNCAPPED, so one huge city/
     phonetic bucket could dump thousands of candidates onto every entity that
     touched it, silently multiplying downstream feature-computation cost.
  3. Both passes (build inverted index, look up candidates) are parallelized across
     cfg["n_jobs"] worker processes via the "fork" start method with
     Pool.imap_unordered, so the shared source DataFrame is inherited via
     copy-on-write (not re-pickled), and only n_jobs chunks are ever in memory
     at once -- not the whole dataset.
"""
from __future__ import annotations
from collections import defaultdict
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


def _cap_for(k: str, cfg: dict) -> int | None:
    if k.startswith(("ntok:", "ng:")):
        return cfg["rare_token_df_max"]
    if k.startswith("atok:"):
        return cfg["addr_token_df_max"]
    if k.startswith(("cs:", "nfirst:", "nfirst2:", "meta:", "init:")):
        return cfg.get("loc_token_df_max", cfg["rare_token_df_max"])
    return None  # pin:/zip: stay uncapped -- exact numeric match, rarely a mega-block


_G_DF = None
_G_CFG = None
_G_INV = None


def _pool_init_scan(df, cfg):
    global _G_DF, _G_CFG
    _G_DF, _G_CFG = df, cfg


def _scan_range(rng: tuple[int, int]) -> dict[str, list[str]]:
    start, end = rng
    df, cfg = _G_DF, _G_CFG
    inv: dict[str, list[str]] = defaultdict(list)
    for row in df.iloc[start:end].itertuples(index=False):
        for k in _keys_row(row, cfg):
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
            cap = _cap_for(k, cfg)
            if cap is not None and len(inv.get(k, ())) > cap:
                continue
            see.update(inv.get(k, ()))
        out[row.entity_id] = sorted(see)
    return out


def build_candidates(s1n: pd.DataFrame, others: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """others = concatenated+normalized S2 & S3. Returns s1_id -> [ids]."""
    n_jobs = cfg.get("n_jobs") or os.cpu_count() or 1
    chunk_size = cfg.get("chunk_size", 200_000)
    ctx = _ctx()

    n_other = len(others)
    if n_jobs > 1 and n_other > chunk_size:
        ranges = list(_chunks_idx(n_other, chunk_size))
        inv: dict[str, list[str]] = defaultdict(list)
        with ctx.Pool(n_jobs, initializer=_pool_init_scan, initargs=(others, cfg)) as pool:
            for part in pool.imap_unordered(_scan_range, ranges):
                for k, v in part.items():
                    inv[k].extend(v)
                del part
    else:
        inv = defaultdict(list)
        for row in others.itertuples(index=False):
            for k in _keys_row(row, cfg):
                inv[k].append(row.entity_id)

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
                cap = _cap_for(k, cfg)
                if cap is not None and len(inv.get(k, ())) > cap:
                    continue
                see.update(inv.get(k, ()))
            out[row.entity_id] = sorted(see)

    return out