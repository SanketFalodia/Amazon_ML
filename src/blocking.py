"""Recall-first candidate generation via a UNION of cheap, complementary blocking keys.

Optimized vs original:
  1. iterrows() -> to_dict("records"): pandas builds all row-dicts in one bulk pass
     instead of constructing a Series per row (the single biggest win on 2M+ rows).
  2. Every key type is now capped (df_max), not just ntok:/ng:/atok:. Previously
     cs: (city+state), nfirst:, nfirst2:, meta:, init: were UNCAPPED, so one huge
     city/phonetic bucket could dump thousands of candidates onto every entity that
     touched it -- silently multiplying downstream feature-computation cost.
  3. Both passes (build inverted index, look up candidates) are parallelized across
     cfg["n_jobs"] processes using the "fork" start method, so large shared objects
     (the inverted index) are inherited via copy-on-write instead of re-pickled.
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


def _keys_row(row: dict, cfg: dict) -> set[str]:
    keys: set[str] = set()
    toks = row["name_no_legal"].split()
    for t in toks:
        keys.add(f"ntok:{t}")
    srt = row["name_sorted"].split()
    if srt:
        keys.add(f"nfirst:{srt[0]}")
        if len(srt) > 1:
            keys.add(f"nfirst2:{srt[0]}_{srt[1]}")
    if row["name_meta"]:
        keys.add(f"meta:{row['name_meta']}")
    if len(row["name_init"]) >= 2:
        keys.add(f"init:{row['name_init']}")
    if row["pincode"]:
        keys.add(f"pin:{row['pincode']}")
    if row["zip5"]:
        keys.add(f"zip:{row['zip5']}")
    if row["city"] and row["state"]:
        keys.add(f"cs:{row['city']}_{row['state']}")
    for t in set(row["addr_no_digits"].split()):
        keys.add(f"atok:{t}")
    comp = row["name_compact"]
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
        # NEW cap -- was previously unlimited. Falls back to rare_token_df_max if
        # you haven't added loc_token_df_max to config.yaml yet.
        return cfg.get("loc_token_df_max", cfg["rare_token_df_max"])
    return None  # pin:/zip: stay uncapped -- exact numeric match, rarely a mega-block


def _build_inv_chunk(records: list[dict], cfg: dict) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = defaultdict(list)
    for r in records:
        for k in _keys_row(r, cfg):
            inv[k].append(r["entity_id"])
    return inv


def _merge_inv(parts: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = defaultdict(list)
    for part in parts:
        for k, v in part.items():
            inv[k].extend(v)
    return inv


_G_INV, _G_CFG = None, None  # populated per-worker by _pool_init, inherited via fork


def _pool_init(inv, cfg):
    global _G_INV, _G_CFG
    _G_INV, _G_CFG = inv, cfg


def _lookup_chunk(records: list[dict]) -> dict[str, list[str]]:
    inv, cfg = _G_INV, _G_CFG
    out: dict[str, list[str]] = {}
    for r in records:
        see: set[str] = set()
        for k in _keys_row(r, cfg):
            cap = _cap_for(k, cfg)
            if cap is not None and len(inv.get(k, ())) > cap:
                continue
            see.update(inv.get(k, ()))
        out[r["entity_id"]] = sorted(see)
    return out


def _chunks(lst: list, n: int):
    size = max(1, -(-len(lst) // n))
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_candidates(s1n: pd.DataFrame, others: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """others = concatenated+normalized S2 & S3. Returns s1_id -> [ids]."""
    n_jobs = cfg.get("n_jobs") or os.cpu_count() or 1
    other_records = others.to_dict("records")
    s1_records = s1n.to_dict("records")
    ctx = _ctx()

    if n_jobs > 1 and len(other_records) > 5000:
        chunks = list(_chunks(other_records, n_jobs))
        with ctx.Pool(n_jobs) as pool:
            parts = pool.starmap(_build_inv_chunk, [(c, cfg) for c in chunks])
        inv = _merge_inv(parts)
    else:
        inv = _build_inv_chunk(other_records, cfg)

    if n_jobs > 1 and len(s1_records) > 5000:
        chunks = list(_chunks(s1_records, n_jobs))
        with ctx.Pool(n_jobs, initializer=_pool_init, initargs=(inv, cfg)) as pool:
            parts = pool.map(_lookup_chunk, chunks)
        out: dict[str, list[str]] = {}
        for p in parts:
            out.update(p)
    else:
        _pool_init(inv, cfg)
        out = _lookup_chunk(s1_records)

    return out