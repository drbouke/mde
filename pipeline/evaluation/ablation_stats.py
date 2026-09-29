"""
Per-fold statistics for the ablation: 95% confidence intervals of mean weighted F1 per condition,
Wilcoxon signed-rank tests of entropy-only against conventional (per dataset over 5 folds and
pooled over all dataset-folds), and the entropy-versus-conventional gap in percentage points.

Usage: python pipeline/evaluation/ablation_stats.py   (writes results/tables/ablation_stats.csv)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import numpy as np
import pandas as pd
from scipy import stats
from config import TABLES

folds = pd.read_csv(TABLES / "ablation_cv_folds.csv")
rows = []
for model in ["LightGBM", "RandomForest"]:
    pooled_conv, pooled_ent = [], []
    for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
        d = folds[(folds.dataset == ds) & (folds.model == model)]
        f = {c: d[d.ablation == c].sort_values("fold").f1.to_numpy() for c in ["conventional", "entropy_only", "combined"]}
        if any(len(v) < 2 for v in f.values()):
            continue
        ci = {c: stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v)) for c, v in f.items()}
        try:
            w = stats.wilcoxon(f["entropy_only"], f["conventional"]).pvalue
        except ValueError:
            w = float("nan")
        try:
            w2 = stats.wilcoxon(f["combined"], f["conventional"]).pvalue
        except ValueError:
            w2 = float("nan")
        pooled_conv += list(f["conventional"]); pooled_ent += list(f["entropy_only"])
        rows.append({"model": model, "dataset": ds,
                     "f1_conv": f["conventional"].mean(), "conv_lo": ci["conventional"][0], "conv_hi": ci["conventional"][1],
                     "f1_ent": f["entropy_only"].mean(), "ent_lo": ci["entropy_only"][0], "ent_hi": ci["entropy_only"][1],
                     "f1_comb": f["combined"].mean(), "comb_lo": ci["combined"][0], "comb_hi": ci["combined"][1],
                     "gap_ent_pp": 100 * (f["conventional"].mean() - f["entropy_only"].mean()),
                     "ent_below_conv_all_folds": bool((f["entropy_only"] < f["conventional"]).all()),
                     "wilcoxon_ent_vs_conv_p": w, "wilcoxon_comb_vs_conv_p": w2})
    if pooled_conv:
        wp = stats.wilcoxon(pooled_ent, pooled_conv).pvalue
        rows.append({"model": model, "dataset": "ALL (pooled folds)", "wilcoxon_ent_vs_conv_p": wp,
                     "gap_ent_pp": 100 * (np.mean(pooled_conv) - np.mean(pooled_ent)),
                     "ent_below_conv_all_folds": bool(np.all(np.array(pooled_ent) < np.array(pooled_conv)))})
out = pd.DataFrame(rows)
out.to_csv(TABLES / "ablation_stats.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
