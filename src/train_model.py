"""Train + calibrate LightGBM on blocking candidates labelled by ground truth."""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from .features import pair_features, feature_columns


def build_training_pairs(s1n, s2n, s3n, cands, truth, name_mat, addr_mat, idx):
    by_id = pd.concat([s2n, s3n]).set_index("entity_id", drop=False)
    X, y, groups, meta = [], [], [], []
    for _, a in s1n.iterrows():
        s1 = a["entity_id"]
        tset = set(truth.get(s1, []))
        for cid in cands.get(s1, []):
            if cid not in by_id.index:
                continue
            b = by_id.loc[cid]
            X.append(pair_features(a, b, name_mat, addr_mat, idx))
            y.append(1 if cid in tset else 0)
            groups.append(s1)
            meta.append((s1, cid))
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
