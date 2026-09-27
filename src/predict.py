"""F0.5-optimal decision: two-threshold search + per-entity greedy Fhat_0.5.

Optimized vs original: score_pairs() gets the same dict-record + multiprocessing
treatment as train_model.build_training_pairs (see that file for why). Everything
below score_pairs (greedy_pick, tune_thresholds, _fhat) is unchanged.
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


_G: dict = {}


def _pool_init(by_id, cands, clf, name_mat, addr_mat, idx):
    _G["by_id"], _G["cands"], _G["clf"] = by_id, cands, clf
    _G["name_mat"], _G["addr_mat"], _G["idx"] = name_mat, addr_mat, idx


def _score_chunk(s1_records: list[dict]):
    by_id, cands, clf = _G["by_id"], _G["cands"], _G["clf"]
    name_mat, addr_mat, idx = _G["name_mat"], _G["addr_mat"], _G["idx"]
    out = {}
    for a in s1_records:
        s1 = a["entity_id"]
        ids = [c for c in cands.get(s1, []) if c in by_id]
        scored = []
        if ids:
            X = pd.DataFrame([pair_features(a, by_id[c], name_mat, addr_mat, idx)
                              for c in ids])[feature_columns()]
            p = clf.predict_proba(X)[:, 1]
            scored = sorted(zip(p.tolist(), ids), reverse=True)
        out[s1] = scored
    return out


def _chunks(lst: list, n: int):
    size = max(1, -(-len(lst) // n))
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def score_pairs(s1n, by_id_df, cands, clf, name_mat, addr_mat, idx,
                n_jobs=None) -> dict[str, list[tuple[float, str]]]:
    by_id = {r["entity_id"]: r for r in by_id_df.to_dict("records")}
    s1_records = s1n.to_dict("records")
    n_jobs = n_jobs or os.cpu_count() or 1
    ctx = _ctx()

    if n_jobs > 1 and len(s1_records) > 2000:
        chunks = list(_chunks(s1_records, n_jobs))
        with ctx.Pool(n_jobs, initializer=_pool_init,
                       initargs=(by_id, cands, clf, name_mat, addr_mat, idx)) as pool:
            parts = pool.map(_score_chunk, chunks)
        out: dict[str, list[tuple[float, str]]] = {}
        for p in parts:
            out.update(p)
        return out

    _pool_init(by_id, cands, clf, name_mat, addr_mat, idx)
    return _score_chunk(s1_records)


def _fhat(probs: list[float]) -> float:
    """Estimated F0.5 of choosing this set given calibrated prob list."""
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