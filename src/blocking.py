"""Recall-first candidate generation via a UNION of cheap, complementary blocking keys."""
from __future__ import annotations
from collections import defaultdict
import pandas as pd


def _keys_row(row, cfg) -> set[str]:
    keys: set[str] = set()
    toks = row["name_no_legal"].split()
    # B1 rare name tokens
    for t in toks:
        keys.add(f"ntok:{t}")
    # B2 sorted first token(s)
    srt = row["name_sorted"].split()
    if srt:
        keys.add(f"nfirst:{srt[0]}")
        if len(srt) > 1:
            keys.add(f"nfirst2:{srt[0]}_{srt[1]}")
    # B4 phonetic
    if row["name_meta"]:
        keys.add(f"meta:{row['name_meta']}")
    # B8 initialism
    if len(row["name_init"]) >= 2:
        keys.add(f"init:{row['name_init']}")
    # B5 pin / zip
    if row["pincode"]:
        keys.add(f"pin:{row['pincode']}")
    if row["zip5"]:
        keys.add(f"zip:{row['zip5']}")
    # B6 city+state
    if row["city"] and row["state"]:
        keys.add(f"cs:{row['city']}_{row['state']}")
    # B7 address tokens
    for t in set(row["addr_no_digits"].split()):
        keys.add(f"atok:{t}")
    # char n-grams of compact name (B3 fallback without LSH lib)
    comp = row["name_compact"]
    n = cfg["char_ngram_n"]
    if len(comp) >= n:
        for i in range(len(comp) - n + 1):
            keys.add(f"ng:{comp[i:i+n]}")
    return keys


def build_candidates(s1n: pd.DataFrame, others: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """others = concatenated+normalized S2 & S3 with an '_src' column. Returns s1_id -> [ids]."""
    inv: dict[str, list[str]] = defaultdict(list)
    for _, r in others.iterrows():
        for k in _keys_row(r, cfg):
            inv[k].append(r["entity_id"])

    df_max = cfg["rare_token_df_max"]
    addr_df_max = cfg["addr_token_df_max"]
    out: dict[str, list[str]] = {}
    for _, r in s1n.iterrows():
        see: set[str] = set()
        for k in _keys_row(r, cfg):
            if k.startswith(("ntok:", "ng:")) and len(inv[k]) > df_max:
                continue
            if k.startswith("atok:") and len(inv[k]) > addr_df_max:
                continue
            see.update(inv.get(k, ()))
        out[r["entity_id"]] = sorted(see)
    return out
