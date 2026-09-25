"""Orchestrator: prep -> block -> featurize -> train -> predict -> validate -> package."""
from __future__ import annotations
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import utils_io as io
from . import normalize as nz
from . import blocking as bl
from . import features as ft
from . import train_model as tm
from . import predict as pr
from .evaluate import macro_f05


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cache(cfg, name):
    d = cfg["paths"]["cache_dir"]
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def stage_prep(cfg):
    tr = cfg["paths"]["train_dir"]
    te = cfg["paths"]["test_dir"]
    s1, s2, s3 = io.read_sources(tr, "train")
    t1, t2, t3 = io.read_sources(te, "test")
    gt = io.read_ground_truth(os.path.join(tr, "train_ground_truth.tsv"))

    all_s1 = list(s1["entity_id"])
    n_single = sum(1 for e in all_s1 if not gt.get(e))
    print(f"[prep] train S1={len(s1)}  S2={len(s2)}  S3={len(s3)}")
    print(f"[prep] singleton fraction f={n_single/len(all_s1):.3f}  "
          f"(all-empty baseline macro-F0.5 ~= f)")

    # cardinality / uniqueness diagnostics
    counts = pd.Series([len(v) for v in gt.values()])
    print("[prep] match-count distribution:\n", counts.value_counts().sort_index().to_string())
    flat = [i for v in gt.values() for i in v]
    dup = len(flat) - len(set(flat))
    print(f"[prep] repeated S2/S3 ids in ground truth = {dup} "
          f"({'many-to-one, uniqueness OK' if dup == 0 else 'NOT one-to-one'})")
    print("[prep] train countries:", s1["country"].value_counts().to_dict())
    print("[prep] test  countries:", t1["country"].value_counts().to_dict())

    s1n, s2n, s3n = nz.normalize_frame(s1), nz.normalize_frame(s2), nz.normalize_frame(s3)
    t1n, t2n, t3n = nz.normalize_frame(t1), nz.normalize_frame(t2), nz.normalize_frame(t3)
    for df, path in ((s1n, "train_s1"), (s2n, "train_s2"), (s3n, "train_s3"),
                     (t1n, "test_s1"), (t2n, "test_s2"), (t3n, "test_s3")):
        df.to_parquet(_cache(cfg, f"{path}.parquet"))
    json.dump(gt, open(_cache(cfg, "ground_truth.json"), "w"))
    print("[prep] done")


def _read_cached(cfg, name):
    return pd.read_parquet(_cache(cfg, f"{name}.parquet"))


def stage_block(cfg):
    s1n = _read_cached(cfg, "train_s1")
    te1 = _read_cached(cfg, "test_s1")
    tr_others = pd.concat([_read_cached(cfg, "train_s2"), _read_cached(cfg, "train_s3")],
                          ignore_index=True)
    te_others = pd.concat([_read_cached(cfg, "test_s2"), _read_cached(cfg, "test_s3")],
                          ignore_index=True)
    tr_cands = bl.build_candidates(s1n, tr_others, cfg["blocking"])
    te_cands = bl.build_candidates(te1, te_others, cfg["blocking"])
    json.dump(tr_cands, open(_cache(cfg, "train_cands.json"), "w"))
    json.dump(te_cands, open(_cache(cfg, "test_cands.json"), "w"))
    io.write_candidates(os.path.join(cfg["paths"]["output_dir"], "candidate_pairs.tsv"), te_cands)
    avg = np.mean([len(v) for v in te_cands.values()])
    print(f"[block] test candidates/entity avg={avg:.1f}  wrote candidate_pairs.tsv")


def stage_featurize(cfg):
    alln = pd.concat([_read_cached(cfg, n) for n in
                      ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3")],
                     ignore_index=True)
    name_mat, addr_mat, idx = ft.build_tfidf(alln)
    import pickle
    with open(_cache(cfg, "vectors.pkl"), "wb") as f:
        pickle.dump((name_mat, addr_mat, idx), f)
    print(f"[feat] tfidf matrices built: name{name_mat.shape} addr{addr_mat.shape}")


def _train_pairs(cfg):
    import pickle
    s1n = _read_cached(cfg, "train_s1")
    s2n = _read_cached(cfg, "train_s2")
    s3n = _read_cached(cfg, "train_s3")
    cands = json.load(open(_cache(cfg, "train_cands.json")))
    gt = json.load(open(_cache(cfg, "ground_truth.json")))
    name_mat, addr_mat, idx = pickle.load(open(_cache(cfg, "vectors.pkl"), "rb"))
    return tm.build_training_pairs(s1n, s2n, s3n, cands, gt, name_mat, addr_mat, idx), (name_mat, addr_mat, idx)


def stage_train(cfg):
    (X, y, groups, meta), _ = _train_pairs(cfg)
    print(f"[train] pairs={len(X)}  positives={int(y.sum())}")
    n_folds = cfg["model"]["n_folds"]
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        clf = tm.train(cfg, X.iloc[tr_idx], y[tr_idx], groups[tr_idx])
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    clf = tm.train(cfg, X, y, groups)                # final model on all data
    tm.save_model(clf, cfg["paths"]["model_path"])
    # tune thresholds on OOF predictions
    scored = _oof_scored(meta, oof)
    all_s1 = list(pd.unique(groups))
    gt = json.load(open(_cache(cfg, "ground_truth.json")))
    best = pr.tune_thresholds(scored, gt, all_s1, cfg)
    json.dump({"t_open": best[1], "t_add": best[2], "oof_f05": best[0]},
              open(_cache(cfg, "thresholds.json"), "w"))
    print(f"[train] OOF macro-F0.5={best[0]:.4f}  t_open={best[1]:.3f}  t_add={best[2]:.3f}")


def _oof_scored(meta, oof):
    scored: dict[str, list[tuple[float, str]]] = {}
    for (s1, cid), p in zip(meta, oof):
        scored.setdefault(s1, []).append((float(p), cid))
    for s1 in scored:
        scored[s1].sort(reverse=True)
    return scored


def stage_predict(cfg):
    import pickle
    clf = tm.load_model(cfg["paths"]["model_path"])
    thr = json.load(open(_cache(cfg, "thresholds.json")))
    te1 = _read_cached(cfg, "test_s1")
    by_id = pd.concat([_read_cached(cfg, "test_s2"), _read_cached(cfg, "test_s3")],
                      ignore_index=True).set_index("entity_id", drop=False)
    cands = json.load(open(_cache(cfg, "test_cands.json")))
    name_mat, addr_mat, idx = pickle.load(open(_cache(cfg, "vectors.pkl"), "rb"))
    scored = pr.score_pairs(te1, by_id, cands, clf, name_mat, addr_mat, idx)
    mm = cfg["decision"].get("max_matches", 0)
    rows = {s1: pr.greedy_pick(sc, thr["t_open"], thr["t_add"], mm) for s1, sc in scored.items()}
    for s1 in te1["entity_id"]:                      # ensure every S1 present
        rows.setdefault(s1, [])
    io.write_results(os.path.join(cfg["paths"]["output_dir"], "matching_results.tsv"), rows)
    print(f"[predict] wrote matching_results.tsv  (t_open={thr['t_open']}, t_add={thr['t_add']})")


def stage_validate(cfg):
    import subprocess
    out = cfg["paths"]["output_dir"]
    cmd = ["python3", "utils/validate_submission.py",
           "--matching", os.path.join(out, "matching_results.tsv"),
           "--candidate", os.path.join(out, "candidate_pairs.tsv"),
           "--test-dir", cfg["paths"]["test_dir"]]
    try:
        print(subprocess.run(cmd, check=False).returncode)
    except FileNotFoundError:
        print("[validate] utils/validate_submission.py not found; running self-check instead")

    te1, te2, te3 = io.read_sources(cfg["paths"]["test_dir"], "test")
    match = _read_out(os.path.join(out, "matching_results.tsv"), "matched_entity_ids")
    cand = _read_out(os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids")
    problems = io.self_check(match, cand, te1, te2, te3)
    print("[validate] self-check:", "OK" if not problems else problems)


def _read_out(path, valcol):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return {r["source1_entity_id"]: [x for x in str(r[valcol]).split(",") if x]
            for _, r in df.iterrows()}


def stage_package(cfg):
    import zipfile
    name = f"{cfg['team_name']}_submission.zip"
    skip = ("__pycache__", ".pyc", ".DS_Store")
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(cfg["paths"]["output_dir"]):
            for fn in files:
                p = os.path.join(root, fn)
                if any(s in p for s in skip):
                    continue
                z.write(p, os.path.relpath(p, "."))
        for root, _, files in os.walk("src"):
            for fn in files:
                p = os.path.join(root, fn)
                if any(s in p for s in skip):
                    continue
                z.write(p, os.path.join("code/business_entity_resolution", p))
        for extra in ("requirements.txt", "config.yaml"):
            if os.path.exists(extra):
                z.write(extra, os.path.join("code/business_entity_resolution", extra))
    print(f"[package] wrote {name}")


def main(argv):
    cfg = load_config()
    stages = {"prep": stage_prep, "block": stage_block, "featurize": stage_featurize,
              "train": stage_train, "predict": stage_predict, "validate": stage_validate,
              "package": stage_package}
    order = ["prep", "block", "featurize", "train", "predict", "validate", "package"]
    todo = order if (not argv or argv[0] == "all") else [argv[0]]
    for s in todo:
        stages[s](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
