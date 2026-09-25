"""Pair features: rapidfuzz string metrics + TF-IDF cosine + phonetic + numeric."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES: list[str] = []


def _tfidf(texts: list[str], key: str):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
    mat = vec.fit_transform(texts)
    return mat, key


def build_tfidf(all_norm: pd.DataFrame):
    name_mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit_transform(
        all_norm["name_no_legal"])
    addr_mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit_transform(
        all_norm["addr_full"])
    ids = {e: i for i, e in enumerate(all_norm["entity_id"])}
    return name_mat, addr_mat, ids


def pair_features(a: pd.Series, b: pd.Series, name_mat, addr_mat, idx) -> dict:
    f = {
        "name_ratio": fuzz.ratio(a["name_no_legal"], b["name_no_legal"]) / 100,
        "name_partial": fuzz.partial_ratio(a["name_no_legal"], b["name_no_legal"]) / 100,
        "name_tsort": fuzz.token_sort_ratio(a["name_no_legal"], b["name_no_legal"]) / 100,
        "name_tset": fuzz.token_set_ratio(a["name_no_legal"], b["name_no_legal"]) / 100,
        "name_wratio": fuzz.WRatio(a["name_no_legal"], b["name_no_legal"]) / 100,
        "name_lev": 1 - distance.Levenshtein.normalized_distance(a["name_no_legal"], b["name_no_legal"]),
        "name_phon": int(bool(a["name_meta"]) and a["name_meta"] == b["name_meta"]),
        "name_init": int(bool(a["name_init"]) and a["name_init"] == b["name_init"]),
        "name_len_ratio": min(len(a["name_no_legal"]), len(b["name_no_legal"]))
                          / (max(len(a["name_no_legal"]), len(b["name_no_legal"])) + 1e-9),
        "addr_ratio": fuzz.ratio(a["addr_full"], b["addr_full"]) / 100,
        "addr_tsort": fuzz.token_sort_ratio(a["addr_no_digits"], b["addr_no_digits"]) / 100,
        "addr_tset": fuzz.token_set_ratio(a["addr_no_digits"], b["addr_no_digits"]) / 100,
        "pin_match": int(bool(a["pincode"]) and a["pincode"] == b["pincode"]),
        "zip_match": int(bool(a["zip5"]) and a["zip5"] == b["zip5"]),
        "house_match": int(bool(a["house_no"]) and a["house_no"] == b["house_no"]),
        "city_match": int(bool(a["city"]) and a["city"] == b["city"]),
        "state_match": int(bool(a["state"]) and a["state"] == b["state"]),
        "country_eq": int(a["country"] == b["country"]),
        "landmark_any": int(bool(a["is_landmark"]) or bool(b["is_landmark"])),
        "src_is_s3": int(str(b["entity_id"]).startswith("S3-")),
    }
    ai, bi = idx.get(a["entity_id"]), idx.get(b["entity_id"])
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
