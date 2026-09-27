"""Train + calibrate LightGBM on blocking candidates labelled by ground truth.

Optimized vs original, in order of impact:
  1. The "b" side (S2+S3 combined, potentially 10M+ rows) is NEVER converted to a
     Python dict/list of records -- it stays a DataFrame indexed by entity_id, and
     individual candidate rows are fetched via .loc[cid] (the same technique the
     ORIGINAL code used, because it's memory-safe: a records-list of this side
     would hold ~10M standalone dict objects simultaneously, often exceeding the
     memory the source parquet files themselves take).
  2. The "a" side (S1 rows, the loop driver) is streamed via itertuples() in
     bounded chunks (default 200k rows) rather than materialized all at once.
  3. Chunks are distributed across cfg-configured worker processes via fork +
     Pool.imap_unordered, so only n_jobs chunks are ever in flight, and the
     shared by_id DataFrame is inherited via copy-on-write, not re-pickled
     per task.
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


def _chunks_idx(n: int, size: int):
    for i in range(0, n, size):
        yield (i, min(i + size, n))


_G: dict = {}


def _pool_init(s1n, by_id, cands, truth, name_mat, addr_mat, idx):
    _G["s1n"], _G["by_id"], _G["cands"], _G["truth"] = s1n, by_id, cands, truth
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _pairs_range(rng: tuple[int, int]):
    start, end = rng
    s1n, by_id, cands, truth = _G["s1n"], _G["by_id"], _G["cands"], _G["truth"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    X, y, groups, meta = [], [], [], []
    for a in s1n.iloc[start:end].itertuples(index=False):
        s1 = a.entity_id
        tset = set(truth.get(s1, []))
        for cid in cands.get(s1, []):
            if cid not in by_id.index:
                continue
            b = by_id.loc[cid]
            X.append(pair_features(a, b, name_mat, addr_mat, idx))
            y.append(1 if cid in tset else 0)
            groups.append(s1)
            meta.append((s1, cid))
    return X, y, groups, meta


def build_training_pairs(s1n, s2n, s3n, cands, truth, name_mat, addr_mat, idx,
                          n_jobs=None, chunk_size=200_000):
    by_id_df = pd.concat([s2n, s3n], ignore_index=True).set_index("entity_id", drop=False)
    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()
    n = len(s1n)

    X, y, groups, meta = [], [], [], []
    if n_jobs > 1 and n > chunk_size:
        ranges = list(_chunks_idx(n, chunk_size))
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(s1n, by_id_df, cands, truth, name_mat, addr_mat, idx)) as pool:
            for px, py, pg, pm in pool.imap_unordered(_pairs_range, ranges):
                X.extend(px)
                y.extend(py)
                groups.extend(pg)
                meta.extend(pm)
                del px, py, pg, pm
    else:
        _pool_init(s1n, by_id_df, cands, truth, name_mat, addr_mat, idx)
        px, py, pg, pm = _pairs_range((0, n))
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