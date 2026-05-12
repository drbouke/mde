# Multi-Level Distributional Entropy (MDE) for Explainable Network Intrusion Detection

Reproducible code for the paper: *Multi-Level Distributional Entropy for Explainable Network Intrusion Detection* (Bouke, 2026).

---

## Repository Layout

```
.
├── src/                        # Core library modules
│   ├── config.py               # Dataset paths, random seeds, ablation settings
│   ├── preprocess.py           # Dataset loading, cleaning, label encoding
│   ├── entropy_features.py     # MDE computation: L1 ADE, L2 JSD, L3 flag entropy
│   ├── fold_pipeline.py        # PercentileClipper; fold-local sklearn Pipeline
│   └── visualize.py            # Figure helpers
│
├── pipeline/                   # Organized experiment entry points
│   ├── experiments/
│   │   ├── run_ablation.py     # Main ablation: 4 datasets × 3 conditions × 2 models
│   │   ├── run_timesplit.py    # CICIDS-2017 temporal split (Mon–Thu → Friday)
│   │   ├── run_temporal_replay.py  # Pseudo-live chronological replay evaluation
│   │   ├── run_baselines.py    # XGBoost + MLP baseline comparison
│   │   ├── run_unseen.py       # Unseen attack family evaluation
│   │   └── run_perclass.py     # Per-category detection rates
│   ├── evaluation/
│   │   └── run_figures.py      # ROC curves, confusion matrices, JSD figure
│   ├── shap/
│   │   └── run_shap.py         # SHAP waterfall/beeswarm + fold stability
│   └── scripts/
│       └── run_all.py          # Top-level orchestrator (runs all stages)
│
├── datasets/                   # Raw data (not tracked in git — see below)
│   ├── NSLKDD/
│   ├── CICIDS2017/
│   ├── CICIDS2018/
│   └── UNSW-NB15/
│
├── results/
│   ├── figures/                # Generated PDFs: ROC, CM, SHAP, ablation
│   └── tables/                 # Generated CSVs: ablation, timesplit, baselines
│
├── requirements.txt
└── README.md
```

---

## Datasets

Download and place each dataset in `datasets/` as follows:

| Dataset | Source | Local path |
|---|---|---|
| NSL-KDD | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | `datasets/NSLKDD/KDD.csv` |
| CICIDS-2017 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | `datasets/CICIDS2017/TrafficLabelling_*.csv` |
| CICIDS-2018 | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) | `datasets/CICIDS2018/datasetcsv.csv` |
| UNSW-NB15 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | `datasets/UNSW-NB15/UNSW-NB15.csv` |

Datasets are not included in the repository due to redistribution restrictions.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. All random seeds are fixed at `RANDOM_STATE = 42` in `src/config.py`.

---

## Reproducing All Results

### Run the full pipeline (recommended)
```bash
python pipeline/scripts/run_all.py
```
This runs all stages in order and writes outputs to `results/tables/` and `results/figures/`.

### Run individual stages
```bash
# Main ablation (5-fold CV, 4 datasets, 3 conditions, LightGBM + RF)
python pipeline/experiments/run_ablation.py

# Temporal generalization (CICIDS-2017 Mon–Thu → Friday, debiased)
python pipeline/experiments/run_timesplit.py

# Pseudo-live temporal replay (chronological windows, fixed vs Youden threshold)
python pipeline/experiments/run_temporal_replay.py

# XGBoost + MLP baselines
python pipeline/experiments/run_baselines.py

# Unseen attack families (Infiltration + Bot held-out)
python pipeline/experiments/run_unseen.py

# Per-category detection rates
python pipeline/experiments/run_perclass.py

# ROC curves, confusion matrices, JSD figure
python pipeline/evaluation/run_figures.py

# SHAP waterfall/beeswarm figures + fold stability metrics
python pipeline/shap/run_shap.py
```

All outputs are written to `results/figures/` (PDFs) and `results/tables/` (CSVs).

---

## Key Design Decisions

**Fold-local preprocessing** — `PercentileClipper` and `SimpleImputer` are fitted inside each CV fold to prevent cross-fold leakage. See `src/fold_pipeline.py`.

**MDE computation** — All entropy features are closed-form scalar functions of each flow's own statistics; no label information is used and no sequence data is required. See `src/entropy_features.py`.

**Debiasing for CICIDS-2017 timesplit** — Features with single-feature AUROC > 0.99 on training data are removed before the temporal evaluation to prevent artifact-driven near-perfect scores.

**Full metric suite** — All experiments report F1, Precision, Recall, DR (binary attack recall), FAR, Accuracy, MCC, ROC-AUC, and PR-AUC. Aggregate F1 alone is insufficient under class imbalance and distribution shift.

---

## Citation

```
@article{bouke2026mde,
  author  = {Bouke, Mohamed Aly},
  title   = {Multi-Level Distributional Entropy for Explainable Network Intrusion Detection},
  year    = {2026},
}
```
