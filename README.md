# Multi-Level Distributional Entropy (MDE) for Explainable Network Intrusion Detection

Reproducible code for the paper **Multi-Level Distributional Entropy for Explainable Network Intrusion Detection** (Bouke et al., 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2606.29797-b31b1b.svg)](https://arxiv.org/abs/2606.29797)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2606.29797-blue.svg)](https://doi.org/10.48550/arXiv.2606.29797)

**Paper:** [arXiv:2606.29797](https://arxiv.org/abs/2606.29797) | **DOI:** [10.48550/arXiv.2606.29797](https://doi.org/10.48550/arXiv.2606.29797)

---

## Repository Layout

```
.
├── src/                        # Core library modules
│   ├── config.py               # Dataset paths, random seeds, ablation settings
│   ├── preprocess.py           # Dataset loading, cleaning, label encoding
│   ├── entropy_features.py     # MDE computation: L1 ADE, L2 JSD, L3 flag entropy
│   ├── fold_pipeline.py        # PercentileClipper, MDEFeatures (fold-local entropy features), Pipelines
│   ├── metrics.py              # Complete metric suite (confusion counts, P/R/F1, DR, FAR, MCC, AUC, PR-AUC)
│   └── visualize.py            # Figure helpers
│
├── pipeline/                   # Organized experiment entry points
│   ├── experiments/
│   │   ├── run_ablation.py     # Main ablation: 4 datasets × 3 conditions × 2 models
│   │   ├── run_timesplit.py    # CICIDS-2017 temporal split (Mon–Thu → Friday)
│   │   ├── run_temporal_replay.py  # Pseudo-live chronological replay evaluation
│   │   ├── run_baselines.py    # Seven-classifier comparison (LGB, RF, XGB, CatBoost, MLP, TabNet, FT-Transformer)
│   │   ├── run_unseen.py       # Unseen attack family evaluation
│   │   ├── run_perclass.py     # Per-category detection rates
│   │   └── run_extra.py        # Cross-dataset transfer, noise robustness, entropy vs simple statistics, profiling
│   ├── evaluation/
│   │   ├── run_figures.py      # ROC curves, confusion matrices, JSD figure
│   │   ├── run_viz.py          # Class distribution, entropy distributions, heatmap
│   │   ├── ablation_stats.py   # Per-fold confidence intervals and Wilcoxon tests
│   │   └── make_tables.py      # LaTeX row blocks for every result table from the CSVs
│   ├── shap/
│   │   ├── run_shap.py         # SHAP waterfall/beeswarm + fold stability
│   │   └── shap_rank_report.py # Mean |SHAP| rankings and top instance contributors
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
│   └── tables/                 # Generated CSVs (summary and per-fold) for every experiment
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

# Seven-classifier comparison
python pipeline/experiments/run_baselines.py

# Unseen attack families (Infiltration + Bot held-out)
python pipeline/experiments/run_unseen.py

# Per-category detection rates
python pipeline/experiments/run_perclass.py

# ROC curves, confusion matrices, JSD figure
python pipeline/evaluation/run_figures.py

# SHAP waterfall/beeswarm figures + fold stability metrics
python pipeline/shap/run_shap.py

# Cross-dataset transfer, noise robustness, entropy vs simple statistics, profiling
python pipeline/experiments/run_extra.py            # or --only profiling (run alone on an idle machine)

# SHAP rankings, per-fold statistics, and LaTeX table rows
python pipeline/shap/shap_rank_report.py
python pipeline/evaluation/ablation_stats.py
python pipeline/evaluation/make_tables.py
```

Every experiment records the complete metric suite for every evaluation set (confusion counts,
accuracy, weighted/macro/attack-class precision, recall and F1, DR, FAR, FNR, TNR, balanced
accuracy, MCC, ROC-AUC, PR-AUC); cross-validated experiments also write a `*_folds.csv` table.

### Notes on the current version
- The cross-directional JSD is computed exactly (64-point Gauss-Hermite quadrature of the two
  Gaussian densities) and lies in [0, ln 2]; `jsd_gaussian_moment_matched` is kept only for comparison.
- CICIDS-2018 as distributed repeats its header line inside the data; the loader removes those rows
  and coerces the affected numeric columns.
- The medians that fill missing entropy inputs and the min-max bounds of the composite score are
  fitted on the training portion of every split (`MDEFeatures` transformer inside the Pipeline).
- Tree ensembles use all cores except in the profiling table, which is single-threaded by definition.
- Every classifier in the comparison table, including MLP, TabNet, and FT-Transformer, trains on the
  full training fold under the same fold-local pipeline. `run_baselines.py` accepts `--datasets`,
  `--models`, and `--suffix` to run a subset (merged with `--merge`), and checkpoints every finished
  fold in `results/tables/_ckpt_baselines.csv` so an interrupted run resumes at the next fold.

All outputs are written to `results/figures/` (PDFs) and `results/tables/` (CSVs).

---

## Key Design Decisions

**Fold-local preprocessing** — `PercentileClipper` and `SimpleImputer` are fitted inside each CV fold to prevent cross-fold leakage. See `src/fold_pipeline.py`.

**MDE computation** — All entropy features are closed-form scalar functions of each flow's own statistics; no label information is used and no sequence data is required. See `src/entropy_features.py`.

**Debiasing for CICIDS-2017 timesplit** — Features with single-feature AUROC > 0.99 on training data are removed before the temporal evaluation to prevent artifact-driven near-perfect scores.

**Full metric suite** — All experiments report F1, Precision, Recall, DR (binary attack recall), FAR, Accuracy, MCC, ROC-AUC, and PR-AUC. Aggregate F1 alone is insufficient under class imbalance and distribution shift.

---

## Citation

```bibtex
@misc{bouke2026multileveldistributionalentropyexplainable,
      title={Multi-Level Distributional Entropy for Explainable Network Intrusion Detection},
      author={Mohamed Aly Bouke and Md Shohel Sayeed and Swee-Huay Heng and Azizol Abdullah and Mohamed Othman},
      year={2026},
      eprint={2606.29797},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      doi={10.48550/arXiv.2606.29797},
      url={https://arxiv.org/abs/2606.29797},
}
```
