"""F0.5-optimal decision: two-threshold search + per-entity greedy Fhat_0.5."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .features import pair_features, feature_columns


def score_pairs(s1n, by_id, cands, clf, name_mat, addr_mat, idx) -> dict[str, list[tuple[float, str]]]:
    out: dict[str, list[tuple[float, str]]] = {}
    for _, a in s1n.iterrows():
        s1 = a["entity_id"]
        scored = []
        ids = [c for c in cands.get(s1, []) if c in by_id.index]
        if ids:
            X = pd.DataFrame([pair_features(a, by_id.loc[c], name_mat, addr_mat, idx)
                              for c in ids])[feature_columns()]
            p = clf.predict_proba(X)[:, 1]
            scored = sorted(zip(p.tolist(), ids), reverse=True)
        out[s1] = scored
    return out


def _fhat(probs: list[float]) -> float:
    """Estimated F0.5 of choosing this set given calibrated prob list."""
    if not probs:
        return 0.0
    k = len(probs)
    esp = sum(probs)               # expected total true matches (denominator for recall est.)
    etp = sum(probs)               # expected true positives of chosen set
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
