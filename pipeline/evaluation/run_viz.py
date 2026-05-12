"""
Regenerate all visualize.py-based figures using only the four target datasets.

Outputs:
  results/figures/class_distribution.pdf
  results/figures/ablation_comparison.pdf
  results/figures/entropy_distribution.pdf
  results/figures/entropy_delta_heatmap.pdf
  results/figures/model_comparison.pdf
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from config import FIGS, TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde
from visualize import (
    plot_class_distribution,
    plot_ablation_comparison,
    plot_entropy_distribution,
    plot_cross_dataset_entropy_heatmap,
    plot_model_comparison,
)

FIGS.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]

# ── Load datasets ────────────────────────────────────────────────────────────
print("Loading datasets...", flush=True)
dfs  = {}
mdes = {}
for name in DATASET_NAMES:
    print(f"  [{name}]", flush=True)
    raw = load_dataset(name)
    df  = clean(raw)
    mde = compute_mde(clean_for_mde(raw), name)
    dfs[name]  = df
    mdes[name] = mde

# ── Figure 1: class distribution ─────────────────────────────────────────────
print("\nPlotting class distribution...", flush=True)
plot_class_distribution(dfs)

# ── Figure 2: entropy distribution (violin) ──────────────────────────────────
print("Plotting entropy distribution...", flush=True)
plot_entropy_distribution(dfs, mdes)

# ── Figure 3: entropy delta heatmap ──────────────────────────────────────────
print("Plotting entropy delta heatmap...", flush=True)
plot_cross_dataset_entropy_heatmap(dfs, mdes)

# ── Figures 4 & 5: ablation comparison + model comparison ────────────────────
ablation_csv = TABLES / "ablation_cv.csv"
if ablation_csv.exists():
    results_df = pd.read_csv(ablation_csv)
    # Keep only the four target datasets
    results_df = results_df[results_df["dataset"].isin(DATASET_NAMES)].reset_index(drop=True)
    print("Plotting ablation comparison...", flush=True)
    plot_ablation_comparison(results_df)
    print("Plotting model comparison...", flush=True)
    plot_model_comparison(results_df)
else:
    print(f"  WARNING: {ablation_csv} not found — skipping ablation/model figures", flush=True)

print("\nDone.", flush=True)
