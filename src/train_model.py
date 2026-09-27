"""Train + calibrate LightGBM on blocking candidates labelled by ground truth.

Optimized vs original:
  1. iterrows()/.loc[] -> plain dicts. Series-level indexing (by_id.loc[cid]) is
     slow one-row-at-a-time; a dict keyed by entity_id is O(1) attribute-style access.
  2. The per-S1-row loop (which drives every pair_features() call -- your actual
     hot path, since it runs ~9 rapidfuzz comparisons + 2 sparse dot products per
     candidate) is split across cfg["n_jobs"] processes via fork, so all cores get used.
"""
from __future__ import annotations
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from .features import pair_features, feature_columns


def _ctx():
    try:
        return get_context("fork")
    except ValueError:
        return get_context()


_G: dict = {}


def _pool_init(by_id, cands, truth, name_mat, addr_mat, idx):
    _G["by_id"], _G["cands"], _G["truth"] = by_id, cands, truth
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _pairs_for_chunk(s1_records: list[dict]):
    by_id, cands, truth = _G["by_id"], _G["cands"], _G["truth"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    X, y, groups, meta = [], [], [], []
    for a in s1_records:
        s1 = a["entity_id"]
        tset = set(truth.get(s1, []))
        for cid in cands.get(s1, []):
            b = by_id.get(cid)
            if b is None:
                continue
            X.append(pair_features(a, b, name_mat, addr_mat, idx))
            y.append(1 if cid in tset else 0)
            groups.append(s1)
            meta.append((s1, cid))
    return X, y, groups, meta


def _chunks(lst: list, n: int):
    size = max(1, -(-len(lst) // n))
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_training_pairs(s1n, s2n, s3n, cands, truth, name_mat, addr_mat, idx, n_jobs=None):
    by_id_df = pd.concat([s2n, s3n], ignore_index=True)
    by_id = {r["entity_id"]: r for r in by_id_df.to_dict("records")}
    s1_records = s1n.to_dict("records")

    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()

    if n_jobs > 1 and len(s1_records) > 2000:
        chunks = list(_chunks(s1_records, n_jobs))
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(by_id, cands, truth, name_mat, addr_mat, idx)) as pool:
            parts = pool.map(_pairs_for_chunk, chunks)
    else:
        _pool_init(by_id, cands, truth, name_mat, addr_mat, idx)
        parts = [_pairs_for_chunk(s1_records)]

    X, y, groups, meta = [], [], [], []
    for px, py, pg, pm in parts:
        X.extend(px)
        y.extend(py)
        groups.extend(pg)
        meta.extend(pm)

    Xdf = pd.DataFrame(X)[feature_columns()]
    return Xdf, np.array(y), np.array(groups), meta


def train(config: dict, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    m = config["model"]
    clf = lgb.LGBMClassifier(
        n_estimators=m["n_estimators"], learning_rate=m["learning_rate"],
        num_leaves=m["num_leaves"], min_child_samples=m["min_child_samples"],
        subsample=m["subsample"], colsample_bytree=m["colsample_bytree"],
        reg_lambda=m["reg_lambda"], n_jobs=-1,
    )
    if m.get("calibrate", True):
        clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    clf.fit(X, y)
    return clf


def save_model(clf, path: str):
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load_model(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)