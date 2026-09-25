"""Macro F0.5 scorer — mirrors the official metric exactly."""
from __future__ import annotations
import numpy as np


def entity_f05(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map: dict[str, list[str]], truth_map: dict[str, list[str]],
              all_s1) -> float:
    scores = [entity_f05(set(pred_map.get(s, [])), set(truth_map.get(s, []))) for s in all_s1]
    return float(np.mean(scores)) if scores else 0.0
