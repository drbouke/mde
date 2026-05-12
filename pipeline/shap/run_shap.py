"""
SHAP analysis: waterfall/beeswarm/bar figures + fold-stability metrics.

Outputs:
  results/figures/shap_waterfall_*.pdf
  results/figures/shap_beeswarm_*.pdf
  results/figures/shap_bar_*.pdf
  results/tables/shap_fold_stability.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import shap
import lightgbm as lgb

from config import FIGS, TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper

FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

DATASETS = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]
MDE_PREFIXES = ("jsd_", "ade_", "dir_entropy", "flag_entropy", "log_",
                "mde_score", "conn_state_entropy", "srv_diversity_entropy",
                "byte_asym_jsd", "ttl_asym_entropy", "log_cv_entropy")
PRIMARY = "#003049"
ACCENT  = "#F77F00"


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )


def is_mde(name):
    return any(name.startswith(p) or name == p for p in MDE_PREFIXES)


def get_shap_explanation(clf, X_sub, feat_names):
    explainer = shap.TreeExplainer(clf)
    exp = explainer(X_sub)
    if exp.values.ndim == 3:
        sv = exp.values[:, :, 1]
        bv = exp.base_values[:, 1] if exp.base_values.ndim > 1 else exp.base_values
        return shap.Explanation(values=sv, base_values=bv,
                                data=exp.data, feature_names=feat_names), sv
    exp.feature_names = feat_names
    return exp, exp.values


def save_waterfall(exp, pos, path, title, top_n=15):
    plt.figure(figsize=(7, max(4, top_n * 0.38)))
    shap.plots.waterfall(exp[pos], max_display=top_n, show=False)
    plt.title(title, fontsize=9, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path.name}", flush=True)


def save_beeswarm(exp, path, title, top_n=15):
    plt.figure(figsize=(7, max(4, top_n * 0.38)))
    shap.plots.beeswarm(exp, max_display=top_n, show=False)
    plt.title(title, fontsize=9, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path.name}", flush=True)


def save_bar(sv, feat_names, path, title, top_n=15):
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
    top = mean_abs.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.35)))
    colors = [ACCENT if any(k in n for k in ["jsd", "ade", "entropy", "mde"])
              else PRIMARY for n in top[::-1].index]
    top[::-1].plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title, fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=ACCENT, label="MDE entropy"),
                       Patch(facecolor=PRIMARY, label="Conventional")],
              fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path.name}", flush=True)


# ── SHAP figures (train on full data) ────────────────────────────────────────
print("Generating SHAP figures...", flush=True)
for ds_name in DATASETS:
    print(f"\n  [{ds_name}]", flush=True)
    raw  = load_dataset(ds_name)
    df   = clean(raw)
    mde  = compute_mde(clean_for_mde(raw), ds_name)
    fsets = build_feature_sets(df, mde)
    X, y, feat = fsets["combined"]

    bad = [f for f in feat if any(k in f.lower()
           for k in ["label", "class", "attack", "binary", "multi"])]
    if bad:
        print(f"  SKIP {ds_name}: label contamination {bad}", flush=True); continue

    clf = make_lgb()
    clf.fit(X, y)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X.shape[0], min(600, X.shape[0]), replace=False)
    X_sub = X[idx]

    exp, sv = get_shap_explanation(clf, X_sub, feat)
    proba = clf.predict_proba(X_sub)[:, 1]
    attack_pos = int(np.argsort(proba)[::-1][0])
    benign_pos = int(np.argsort(proba)[0])
    tag = ds_name.replace("-", "_").replace(" ", "_")

    save_waterfall(exp, attack_pos,
                   FIGS / f"shap_waterfall_{tag}_LightGBM_combined_attack.pdf",
                   f"SHAP Waterfall (Attack) — {ds_name}")
    save_waterfall(exp, benign_pos,
                   FIGS / f"shap_waterfall_{tag}_LightGBM_combined_benign.pdf",
                   f"SHAP Waterfall (Benign) — {ds_name}")
    save_beeswarm(exp,
                  FIGS / f"shap_beeswarm_{tag}_LightGBM_combined.pdf",
                  f"SHAP Beeswarm — {ds_name}")
    save_bar(sv, feat,
             FIGS / f"shap_bar_{tag}_LightGBM_combined.pdf",
             f"SHAP Importance — {ds_name}")

# ── SHAP fold stability ───────────────────────────────────────────────────────
print("\nComputing SHAP fold stability...", flush=True)
stability_rows = []
for ds_name in ["NSL-KDD", "CICIDS-2017", "UNSW-NB15"]:
    print(f"\n  [{ds_name}]", flush=True)
    raw  = load_dataset(ds_name)
    df   = clean(raw)
    mde  = compute_mde(clean_for_mde(raw), ds_name)
    fsets = build_feature_sets(df, mde)
    X, y, feat = fsets["combined"]
    feat = list(feat)
    mde_idx = [i for i, f in enumerate(feat) if is_mde(f)]

    fold_imps = []
    for fold_i, (tr, te) in enumerate(CV.split(X, y)):
        X_tr, X_te, y_tr = X[tr], X[te], y[tr]
        imp = SimpleImputer(strategy="median")
        cli = PercentileClipper()
        X_tr = cli.fit_transform(imp.fit_transform(X_tr))
        X_te = cli.transform(imp.transform(X_te))
        clf  = make_lgb(); clf.fit(X_tr, y_tr)
        rng  = np.random.RandomState(RANDOM_STATE + fold_i)
        n_sh = min(300, len(X_te))
        X_sh = X_te[rng.choice(len(X_te), n_sh, replace=False)]
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_sh)
        if isinstance(sv, list): sv = sv[1]
        elif sv.ndim == 3:       sv = sv[:, :, 1]
        fold_imps.append(np.abs(sv).mean(axis=0))
        print(f"    fold {fold_i+1}/5 done", flush=True)

    fold_imps = np.array(fold_imps)
    rhos_all, rhos_mde, taus_all = [], [], []
    for i, j in combinations(range(5), 2):
        rho, _ = spearmanr(fold_imps[i], fold_imps[j])
        tau, _ = kendalltau(fold_imps[i], fold_imps[j])
        rhos_all.append(rho); taus_all.append(tau)
        if len(mde_idx) >= 2:
            rho_m, _ = spearmanr(fold_imps[i, mde_idx], fold_imps[j, mde_idx])
            rhos_mde.append(rho_m)

    ranks = np.argsort(np.argsort(-fold_imps, axis=1), axis=1)
    mde_rank_std = ranks[:, mde_idx].std(axis=0) if mde_idx else np.array([])

    print(f"  Spearman all={np.mean(rhos_all):.3f}±{np.std(rhos_all):.3f}  "
          f"MDE={np.mean(rhos_mde):.3f}±{np.std(rhos_mde):.3f}  "
          f"tau={np.mean(taus_all):.3f}", flush=True)

    stability_rows.append({
        "dataset":           ds_name,
        "n_mde_feats":       len(mde_idx),
        "spearman_all_mean": round(np.mean(rhos_all), 3),
        "spearman_all_std":  round(np.std(rhos_all),  3),
        "spearman_mde_mean": round(np.mean(rhos_mde) if rhos_mde else np.nan, 3),
        "spearman_mde_std":  round(np.std(rhos_mde)  if rhos_mde else np.nan, 3),
        "kendall_all_mean":  round(np.mean(taus_all), 3),
        "kendall_all_std":   round(np.std(taus_all),  3),
        "mde_rank_std_mean": round(mde_rank_std.mean(), 2) if len(mde_rank_std) else np.nan,
    })

df_stab = pd.DataFrame(stability_rows)
df_stab.to_csv(TABLES / "shap_fold_stability.csv", index=False)
print(f"\nSaved: {TABLES / 'shap_fold_stability.csv'}", flush=True)
print(df_stab.to_string(index=False), flush=True)
print("\nDone.", flush=True)
