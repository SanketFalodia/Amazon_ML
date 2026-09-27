"""F0.5-optimal decision: two-threshold search + per-entity greedy Fhat_0.5.

score_pairs() takes s2n/s3n SEPARATELY and dispatches by entity_id prefix
(S2- / S3-), same reasoning as train_model.build_training_pairs: avoids the
pd.concat() memory spike on multi-million-row frames. Everything below
score_pairs (greedy_pick, tune_thresholds, _fhat) is unchanged.
"""
from __future__ import annotations
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

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


def _pool_init(s1n, s2_idx, s3_idx, cands, clf, name_mat, addr_mat, idx):
    _G["s1n"], _G["s2_idx"], _G["s3_idx"] = s1n, s2_idx, s3_idx
    _G["cands"], _G["clf"] = cands, clf
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _score_range(rng: tuple[int, int]):
    start, end = rng
    s1n, s2_idx, s3_idx = _G["s1n"], _G["s2_idx"], _G["s3_idx"]
    cands, clf = _G["cands"], _G["clf"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    out = {}
    for a in s1n.iloc[start:end].itertuples(index=False):
        s1 = a.entity_id
        rows, cids = [], []
        for cid in cands.get(s1, []):
            b = _lookup_b(cid, s2_idx, s3_idx)
            if b is None:
                continue
            rows.append(pair_features(a, b, name_mat, addr_mat, idx))
            cids.append(cid)
        scored = []
        if rows:
            X = pd.DataFrame(rows)[feature_columns()]
            p = clf.predict_proba(X)[:, 1]
            scored = sorted(zip(p.tolist(), cids), reverse=True)
        out[s1] = scored
    return out


def score_pairs(s1n, s2n, s3n, cands, clf, name_mat, addr_mat, idx,
                n_jobs=None, chunk_size=200_000) -> dict[str, list[tuple[float, str]]]:
    s2_idx = s2n.set_index("entity_id", drop=False)
    s3_idx = s3n.set_index("entity_id", drop=False)
    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()
    n = len(s1n)

    if n_jobs > 1 and n > chunk_size:
        ranges = list(_chunks_idx(n, chunk_size))
        out: dict[str, list[tuple[float, str]]] = {}
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(s1n, s2_idx, s3_idx, cands, clf, name_mat, addr_mat, idx)) as pool:
            for part in pool.imap_unordered(_score_range, ranges):
                out.update(part)
                del part
        return out

    _pool_init(s1n, s2_idx, s3_idx, cands, clf, name_mat, addr_mat, idx)
    return _score_range((0, n))


def _fhat(probs: list[float]) -> float:
    if not probs:
        return 0.0
    k = len(probs)
    esp = sum(probs)
    etp = sum(probs)
    if esp <= 0 or etp <= 0:
        return 0.0
    prec = etp / k
    rec = etp / esp
    return 1.25 * prec * rec / (0.25 * prec + rec)


def greedy_pick(scored: list[tuple[float, str]], t_open: float, t_add: float,
                max_matches: int = 0) -> list[str]:
    if not scored or scored[0][0] < t_open:
        return []
    picks = [scored[0][1]]
    best = _fhat([scored[0][0]])
    probs = [scored[0][0]]
    for p, cid in scored[1:]:
        if p < t_add:
            break
        cand_probs = probs + [p]
        cand = _fhat(cand_probs)
        if cand >= best:
            best = cand
            probs = cand_probs
            picks.append(cid)
        if max_matches and len(picks) >= max_matches:
            break
    return picks


def tune_thresholds(scored, truth, all_s1, cfg) -> tuple[float, float, float]:
    from .evaluate import macro_f05
    s0, s1_, step = cfg["decision"]["t_open_grid"]
    a0, a1, astep = cfg["decision"]["t_add_grid"]
    mm = cfg["decision"].get("max_matches", 0)
    greedy = cfg["decision"].get("greedy_per_entity", True)

    best = (-1.0, 0.5, 0.3)
    for t_open in np.arange(s0, s1_ + 1e-9, step):
        for t_add in np.arange(a0, t_open + 1e-9, astep):
            pred = {}
            for s1 in all_s1:
                sc = scored.get(s1, [])
                if greedy:
                    pred[s1] = greedy_pick(sc, t_open, t_add, mm)
                else:
                    picks = [cid for p, cid in sc if p >= t_add]
                    if sc and sc[0][0] >= t_open:
                        pred[s1] = picks
                    else:
                        pred[s1] = []
            f = macro_f05(pred, truth, all_s1)
            if f > best[0]:
                best = (f, float(t_open), float(t_add))
    return best