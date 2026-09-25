"""TSV IO helpers + format self-checks. All challenge files are tab-separated."""
from __future__ import annotations
import os
import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    # CRITICAL: always use an explicit tab separator.
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def read_sources(dirpath: str, split: str):
    """split in {'train','test'} -> (s1, s2, s3)."""
    s1 = read_tsv(os.path.join(dirpath, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(dirpath, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(dirpath, f"{split}_source3.tsv"))
    return s1, s2, s3


def read_ground_truth(path: str) -> dict[str, list[str]]:
    gt = read_tsv(path)
    out: dict[str, list[str]] = {}
    for _, r in gt.iterrows():
        ids = [x for x in str(r["matched_entity_ids"]).split(",") if x.strip()]
        out[r["source1_entity_id"]] = ids
    return out


def _fmt(ids) -> str:
    return ",".join(ids)


def write_results(path: str, rows: dict[str, list[str]]) -> None:
    """rows: s1_id -> list of ids (possibly empty). One row per S1, sorted."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["source1_entity_id\tmatched_entity_ids"]
    for s1 in sorted(rows):
        lines.append(f"{s1}\t{_fmt(rows[s1])}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_candidates(path: str, rows: dict[str, list[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for s1 in sorted(rows):
        lines.append(f"{s1}\t{_fmt(rows[s1])}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def self_check(matching: dict[str, list[str]], candidates: dict[str, list[str]],
               test_s1: pd.DataFrame, test_s2: pd.DataFrame, test_s3: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty == ok)."""
    problems: list[str] = []
    valid_ids = set(test_s2["entity_id"]) | set(test_s3["entity_id"])
    all_s1 = set(test_s1["entity_id"])

    if set(matching) != all_s1:
        missing = all_s1 - set(matching)
        extra = set(matching) - all_s1
        if missing:
            problems.append(f"missing {len(missing)} S1 entities in matching")
        if extra:
            problems.append(f"{len(extra)} unknown S1 entities in matching")

    for s1, ids in matching.items():
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate ids in list for {s1}")
        bad = [i for i in ids if i not in valid_ids]
        if bad:
            problems.append(f"{s1}: ids not in test set -> {bad[:3]}")
        cand = set(candidates.get(s1, []))
        if not set(ids) <= cand:
            problems.append(f"{s1}: match not present in candidate_pairs")
    return problems
