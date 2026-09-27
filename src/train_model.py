%%writefile src/train_model.py
"""Train + calibrate LightGBM on blocking candidates labelled by ground truth.

build_training_pairs() takes s2n/s3n SEPARATELY and dispatches by entity_id
prefix (S2-/S3-) via _lookup_b, instead of pd.concat()-ing them into one
by_id frame first -- same pattern, same reasoning, as predict.score_pairs
(see that module's docstring). At train scale (s2n+s3n potentially 10M+ rows
combined) a concat briefly holds both source frames AND the concatenated
result simultaneously, which is exactly the kind of spike the rest of this
codebase (blocking.py, features.py, stage_block) already avoids -- this
mirrors that here too.

The "a" side (S1 rows, the loop driver -- already bounded by the pair-budget
sample in pipeline._sample_s1_by_pair_budget, so this never sees the full
2M+ row S1 set) streams via itertuples() in bounded chunks distributed
across worker processes via fork + Pool.imap_unordered, same as before.
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


def _lookup_b(cid: str, s2_idx: pd.DataFrame, s3_idx: pd.DataFrame):
    if cid.startswith("S2-"):
        return s2_idx.loc[cid] if cid in s2_idx.index else None
    if cid.startswith("S3-"):
        return s3_idx.loc[cid] if cid in s3_idx.index else None
    return None


_G: dict = {}


def _pool_init(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx):
    _G["s1n"], _G["s2_idx"], _G["s3_idx"] = s1n, s2_idx, s3_idx
    _G["cands"], _G["truth"] = cands, truth
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _pairs_range(rng: tuple[int, int]):
    start, end = rng
    s1n, s2_idx, s3_idx = _G["s1n"], _G["s2_idx"], _G["s3_idx"]
    cands, truth = _G["cands"], _G["truth"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    X, y, groups, meta = [], [], [], []
    for a in s1n.iloc[start:end].itertuples(index=False):
        s1 = a.entity_id
        tset = set(truth.get(s1, []))
        for cid in cands.get(s1, []):
            b = _lookup_b(cid, s2_idx, s3_idx)
            if b is None:
                continue
            X.append(pair_features(a, b, name_mat, addr_mat, idx))
            y.append(1 if cid in tset else 0)
            groups.append(s1)
            meta.append((s1, cid))
    return X, y, groups, meta


def build_training_pairs(s1n, s2n, s3n, cands, truth, name_mat, addr_mat, idx,
                          n_jobs=None, chunk_size=200_000):
    s2_idx = s2n.set_index("entity_id", drop=False)
    s3_idx = s3n.set_index("entity_id", drop=False)
    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()
    n = len(s1n)

    X, y, groups, meta = [], [], [], []
    if n_jobs > 1 and n > chunk_size:
        ranges = list(_chunks_idx(n, chunk_size))
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx)) as pool:
            for px, py, pg, pm in pool.imap_unordered(_pairs_range, ranges):
                X.extend(px)
                y.extend(py)
                groups.extend(pg)
                meta.extend(pm)
                del px, py, pg, pm
    else:
        _pool_init(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx)
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
        return pickle.load(f)"""Train + calibrate LightGBM on blocking candidates labelled by ground truth.

build_training_pairs() takes s2n/s3n SEPARATELY and dispatches by entity_id
prefix (S2-/S3-) via _lookup_b, instead of pd.concat()-ing them into one
by_id frame first -- same pattern, same reasoning, as predict.score_pairs
(see that module's docstring). At train scale (s2n+s3n potentially 10M+ rows
combined) a concat briefly holds both source frames AND the concatenated
result simultaneously, which is exactly the kind of spike the rest of this
codebase (blocking.py, features.py, stage_block) already avoids -- this
mirrors that here too.

The "a" side (S1 rows, the loop driver -- already bounded by the pair-budget
sample in pipeline._sample_s1_by_pair_budget, so this never sees the full
2M+ row S1 set) streams via itertuples() in bounded chunks distributed
across worker processes via fork + Pool.imap_unordered, same as before.
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


def _lookup_b(cid: str, s2_idx: pd.DataFrame, s3_idx: pd.DataFrame):
    if cid.startswith("S2-"):
        return s2_idx.loc[cid] if cid in s2_idx.index else None
    if cid.startswith("S3-"):
        return s3_idx.loc[cid] if cid in s3_idx.index else None
    return None


_G: dict = {}


def _pool_init(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx):
    _G["s1n"], _G["s2_idx"], _G["s3_idx"] = s1n, s2_idx, s3_idx
    _G["cands"], _G["truth"] = cands, truth
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _pairs_range(rng: tuple[int, int]):
    start, end = rng
    s1n, s2_idx, s3_idx = _G["s1n"], _G["s2_idx"], _G["s3_idx"]
    cands, truth = _G["cands"], _G["truth"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    X, y, groups, meta = [], [], [], []
    for a in s1n.iloc[start:end].itertuples(index=False):
        s1 = a.entity_id
        tset = set(truth.get(s1, []))
        for cid in cands.get(s1, []):
            b = _lookup_b(cid, s2_idx, s3_idx)
            if b is None:
                continue
            X.append(pair_features(a, b, name_mat, addr_mat, idx))
            y.append(1 if cid in tset else 0)
            groups.append(s1)
            meta.append((s1, cid))
    return X, y, groups, meta


def build_training_pairs(s1n, s2n, s3n, cands, truth, name_mat, addr_mat, idx,
                          n_jobs=None, chunk_size=200_000):
    s2_idx = s2n.set_index("entity_id", drop=False)
    s3_idx = s3n.set_index("entity_id", drop=False)
    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()
    n = len(s1n)

    X, y, groups, meta = [], [], [], []
    if n_jobs > 1 and n > chunk_size:
        ranges = list(_chunks_idx(n, chunk_size))
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx)) as pool:
            for px, py, pg, pm in pool.imap_unordered(_pairs_range, ranges):
                X.extend(px)
                y.extend(py)
                groups.extend(pg)
                meta.extend(pm)
                del px, py, pg, pm
    else:
        _pool_init(s1n, s2_idx, s3_idx, cands, truth, name_mat, addr_mat, idx)
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