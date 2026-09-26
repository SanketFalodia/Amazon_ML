# Business Entity Resolution — Amazon ML Challenge 2026 starter kit

A config-driven, cache-friendly skeleton implementing the playbook:
**normalize → block → featurize → train → predict (F0.5-optimal) → validate → package**.

> Only uses the provided data and a permissively licensed model (LightGBM, MIT). No external lookups.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Put the data where the config expects it

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
  ├── test_source1.tsv
  ├── test_source2.tsv
  └── test_source3.tsv
utils/validate_submission.py        # provided by the organizers
```

Each of the six source TSV files must be tab-separated and contain these columns:

```text
entity_id	business_name	business_address	country
```

For your three test files, place them in `dataset/test/` in this order: source 1 is
the S1 query table, while sources 2 and 3 are the candidate entity tables. If your
filenames differ, update `paths.test_sources` in `config.yaml`; it must contain
exactly three filenames.

## 3. Run everything

```bash
make run          # prep -> block -> featurize -> train -> predict -> validate -> package
```

or step by step:

```bash
python -m src.pipeline prep        # normalize + cache Parquet
python -m src.pipeline block       # build candidates -> output/candidate_pairs.tsv
python -m src.pipeline featurize   # build pair features (cached)
python -m src.pipeline train       # train + calibrate LightGBM, save model
python -m src.pipeline predict     # two-threshold F0.5 decision -> output/matching_results.tsv
python -m src.pipeline validate    # run the organizers' validator
python -m src.pipeline package     # build <team>_submission.zip
```

Everything is driven by `config.yaml` — change the YAML, not the code, for fast iteration.
Intermediate artefacts are cached under `cache/`, so re-runs of a single stage are instant.

## 4. Tune for F0.5

- `T_open` (must-open threshold for a singleton to get a match) and `T_add` (threshold to add extra matches)
  are grid-searched on a GroupKFold-by-S1 validation split inside `predict.py`.
- Watch the printed `macro F0.5` — that is the leaderboard metric.
- Compare against the **all-empty floor** = singleton fraction (printed in `prep`).

## 5. Layout

```
src/normalize.py   -- multi-view name/address/country normalization
src/blocking.py    -- union-of-keys recall-first candidate generation
src/features.py    -- rapidfuzz + TF-IDF + phonetic + numeric pair features
src/train_model.py -- calibrated LightGBM on hard-negative-mined pairs
src/predict.py     -- two-threshold + per-entity greedy Fhat_0.5 decision
src/evaluate.py    -- macro F0.5 scorer (mirrors the official metric)
src/utils_io.py    -- TSV IO + self-checks
src/pipeline.py    -- orchestrator
```
