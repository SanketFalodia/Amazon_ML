"""Pair features: rapidfuzz string metrics + TF-IDF cosine + phonetic + numeric.

Changes vs original:
  - _v(row, key): generic field accessor so pair_features can take a namedtuple
    (from itertuples, used for the S1/"a" side) and a pandas Series (from .loc[],
    used for the S2+S3/"b" side) interchangeably.
  - build_tfidf(dfs, ...): accepts a LIST of dataframes (e.g. all 6 cached
    parquet frames) and builds the name/addr/entity_id lists directly via
    per-frame .tolist() + list.extend(), instead of pd.concat()-ing all of
    them into one ~22M-row, 21-column frame first. Only 3 lists of plain
    strings are ever held (not a full duplicated dataframe), and
    max_features bounds the resulting vocabulary size.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES: list[str] = []


def _v(row, key):
    """Works for namedtuples (itertuples -> attribute access) and for pandas
    Series / dicts (-> item access), so callers never need to pick one
    representation for both sides of a pair."""
    try:
        return getattr(row, key)
    except AttributeError:
        return row[key]


def build_tfidf(dfs, max_features: int = 100_000):
    """dfs: a DataFrame OR a list/tuple of DataFrames. Never concatenated --
    only the three needed columns are extracted and extended into plain
    lists, which is far cheaper than pd.concat() on the full ~21-column
    frames at multi-million-row scale."""
    frames = list(dfs) if isinstance(dfs, (list, tuple)) else [dfs]

    name_texts: list[str] = []
    addr_texts: list[str] = []
    entity_ids: list[str] = []
    for df in frames:
        name_texts.extend(df["name_no_legal"].tolist())
        addr_texts.extend(df["addr_full"].tolist())
        entity_ids.extend(df["entity_id"].tolist())

    name_mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                max_features=max_features).fit_transform(name_texts)
    addr_mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                max_features=max_features).fit_transform(addr_texts)
    idx = {e: i for i, e in enumerate(entity_ids)}
    return name_mat, addr_mat, idx


def pair_features(a, b, name_mat, addr_mat, idx) -> dict:
    an, bn = _v(a, "name_no_legal"), _v(b, "name_no_legal")
    aa, ba = _v(a, "addr_full"), _v(b, "addr_full")
    aad, bad = _v(a, "addr_no_digits"), _v(b, "addr_no_digits")
    f = {
        "name_ratio": fuzz.ratio(an, bn) / 100,
        "name_partial": fuzz.partial_ratio(an, bn) / 100,
        "name_tsort": fuzz.token_sort_ratio(an, bn) / 100,
        "name_tset": fuzz.token_set_ratio(an, bn) / 100,
        "name_wratio": fuzz.WRatio(an, bn) / 100,
        "name_lev": 1 - distance.Levenshtein.normalized_distance(an, bn),
        "name_phon": int(bool(_v(a, "name_meta")) and _v(a, "name_meta") == _v(b, "name_meta")),
        "name_init": int(bool(_v(a, "name_init")) and _v(a, "name_init") == _v(b, "name_init")),
        "name_len_ratio": min(len(an), len(bn)) / (max(len(an), len(bn)) + 1e-9),
        "addr_ratio": fuzz.ratio(aa, ba) / 100,
        "addr_tsort": fuzz.token_sort_ratio(aad, bad) / 100,
        "addr_tset": fuzz.token_set_ratio(aad, bad) / 100,
        "pin_match": int(bool(_v(a, "pincode")) and _v(a, "pincode") == _v(b, "pincode")),
        "zip_match": int(bool(_v(a, "zip5")) and _v(a, "zip5") == _v(b, "zip5")),
        "house_match": int(bool(_v(a, "house_no")) and _v(a, "house_no") == _v(b, "house_no")),
        "city_match": int(bool(_v(a, "city")) and _v(a, "city") == _v(b, "city")),
        "state_match": int(bool(_v(a, "state")) and _v(a, "state") == _v(b, "state")),
        "country_eq": int(_v(a, "country") == _v(b, "country")),
        "landmark_any": int(bool(_v(a, "is_landmark")) or bool(_v(b, "is_landmark"))),
        "src_is_s3": int(str(_v(b, "entity_id")).startswith("S3-")),
    }
    ai, bi = idx.get(_v(a, "entity_id")), idx.get(_v(b, "entity_id"))
    if ai is not None and bi is not None:
        f["name_tfidf"] = float(name_mat[ai].multiply(name_mat[bi]).sum())
        f["addr_tfidf"] = float(addr_mat[ai].multiply(addr_mat[bi]).sum())
    else:
        f["name_tfidf"] = f["addr_tfidf"] = 0.0
    return f


def feature_columns() -> list[str]:
    return [
        "name_ratio", "name_partial", "name_tsort", "name_tset", "name_wratio", "name_lev",
        "name_phon", "name_init", "name_len_ratio", "addr_ratio", "addr_tsort", "addr_tset",
        "pin_match", "zip_match", "house_match", "city_match", "state_match", "country_eq",
        "landmark_any", "src_is_s3", "name_tfidf", "addr_tfidf",
    ]