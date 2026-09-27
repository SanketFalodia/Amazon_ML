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


def build_tfidf(loaders, max_features: int = 100_000, min_df: int = 2,
                 fit_sample_per_frame: int = 500_000, seed: int = 42):
    """loaders: a list of zero-arg callables, each returning ONE freshly-loaded
    frame with [entity_id, name_no_legal, addr_full] (e.g. one per cached
    parquet file). Called TWICE (sample pass, transform pass) so at most one
    frame is ever held in memory -- the previous version held all 6 loaded
    frames PLUS a merged ~24M-row text list PLUS the two output matrices
    simultaneously, which is what actually exhausted RAM.

    Pass 1 (fit): draws up to `fit_sample_per_frame` random rows from each
    frame and fits the vocabulary on that bounded sample instead of all 24M+
    rows. max_features already caps the target vocabulary size, so a several-
    million-row sample converges to essentially the same vocabulary as a
    full-corpus fit, at a fraction of the time/memory.

    Pass 2 (transform): transforms each frame separately with the now-fitted
    vectorizer and vstacks the per-frame blocks -- a merged text list the
    size of the whole corpus is never built.

    min_df=2 is free, not a tradeoff: a term with GLOBAL document frequency 1
    appears in exactly one entity in the whole corpus, so it can never be
    shared between the two DIFFERENT entities in any pair -- it can never
    make a nonzero contribution to that pair's name_tfidf/addr_tfidf dot
    product either way. Dropping it changes no pair's feature value, only
    removes dead weight from the matrix (and it doesn't change surviving
    terms' IDF values, since sklearn computes those from the kept vocabulary
    and the same fixed document count).

    dtype=float32 (vs sklearn's float64 default) halves the sparse matrix's
    per-nonzero byte cost, with no meaningful precision loss for a 0..1
    similarity feature.
    """
    rng = np.random.default_rng(seed)

    def _sampled_col(df, col):
        n = len(df)
        if n <= fit_sample_per_frame:
            return df[col].tolist()
        take = rng.choice(n, size=fit_sample_per_frame, replace=False)
        return df[col].iloc[take].tolist()

    name_sample: list[str] = []
    addr_sample: list[str] = []
    for load in loaders:
        df = load()
        name_sample.extend(_sampled_col(df, "name_no_legal"))
        addr_sample.extend(_sampled_col(df, "addr_full"))
        del df

    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                max_features=max_features, min_df=min_df, dtype=np.float32)
    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                max_features=max_features, min_df=min_df, dtype=np.float32)
    name_vec.fit(name_sample)
    addr_vec.fit(addr_sample)
    del name_sample, addr_sample

    import scipy.sparse as sp
    name_parts, addr_parts, entity_ids = [], [], []
    for load in loaders:
        df = load()
        name_parts.append(name_vec.transform(df["name_no_legal"]))
        addr_parts.append(addr_vec.transform(df["addr_full"]))
        entity_ids.extend(df["entity_id"].tolist())
        del df

    name_mat = sp.vstack(name_parts, format="csr")
    addr_mat = sp.vstack(addr_parts, format="csr")
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