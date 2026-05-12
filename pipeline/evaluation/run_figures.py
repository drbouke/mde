"""
Generate ROC curves, confusion matrices, and JSD empirical distribution figure.

Outputs:
  results/figures/roc_all_conditions.pdf
  results/figures/cm_all_datasets.pdf
  results/figures/jsd_empirical_dist.pdf
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from config import FIGS, TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper

FIGS.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

DATASETS    = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]
CONDITIONS  = ["conventional", "entropy_only", "combined"]
COND_LABELS = {"conventional": "Conventional", "entropy_only": "Entropy-only", "combined": "Combined"}
COND_COLORS = {"conventional": "#4878d0", "entropy_only": "#ee854a", "combined": "#6acc65"}
COND_LS     = {"conventional": "-", "entropy_only": "--", "combined": "-."}
PRIMARY = "#003049"
ACCENT  = "#F77F00"
LN2     = np.log(2)


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )


def fold_roc(X, y):
    base_fpr = np.linspace(0, 1, 201)
    tprs, aucs = [], []
    for tr, te in CV.split(X, y):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        imp = SimpleImputer(strategy="median").fit(X_tr)
        cli = PercentileClipper().fit(imp.transform(X_tr))
        Xtr = cli.transform(imp.transform(X_tr))
        Xte = cli.transform(imp.transform(X_te))
        clf = make_lgb()
        clf.fit(Xtr, y_tr)
        fpr, tpr, _ = roc_curve(y_te, clf.predict_proba(Xte)[:, 1])
        tpr_i = np.interp(base_fpr, fpr, tpr); tpr_i[0] = 0.0
        tprs.append(tpr_i); aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    return base_fpr, mean_tpr, np.mean(aucs)


def fold_cm(X, y):
    cm_total = np.zeros((2, 2), dtype=int)
    for tr, te in CV.split(X, y):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        imp = SimpleImputer(strategy="median").fit(X_tr)
        cli = PercentileClipper().fit(imp.transform(X_tr))
        Xtr = cli.transform(imp.transform(X_tr))
        Xte = cli.transform(imp.transform(X_te))
        clf = make_lgb()
        clf.fit(Xtr, y_tr)
        cm_total += confusion_matrix(y_te, clf.predict(Xte), labels=[0, 1])
    return cm_total


# ── Load all datasets ────────────────────────────────────────────────────────
print("Loading datasets...", flush=True)
fsets_all = {}
for ds_name in DATASETS:
    print(f"  [{ds_name}]", flush=True)
    raw = load_dataset(ds_name)
    df  = clean(raw)
    mde = compute_mde(clean_for_mde(raw), ds_name)
    fsets_all[ds_name] = build_feature_sets(df, mde)

# ── ROC curves ───────────────────────────────────────────────────────────────
print("\nGenerating ROC curves...", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
fig.subplots_adjust(wspace=0.32, left=0.05, right=0.98, top=0.88, bottom=0.16)
for col, ds_name in enumerate(DATASETS):
    ax = axes[col]
    for cond in CONDITIONS:
        X, y, _ = fsets_all[ds_name][cond]
        print(f"    {ds_name} | {cond}", flush=True)
        fpr, tpr, roc_auc = fold_roc(X, y)
        ax.plot(fpr, tpr, color=COND_COLORS[cond], ls=COND_LS[cond], lw=1.8,
                label=f"{COND_LABELS[cond]} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=8)
    if col == 0:
        ax.set_ylabel("True Positive Rate", fontsize=8)
    ax.set_title(ds_name, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.25, lw=0.5)
fig.suptitle("ROC Curves — Four Datasets and Feature Conditions (LightGBM, 5-fold CV)",
             fontsize=10, fontweight="bold", y=0.98)
out_roc = FIGS / "roc_all_conditions.pdf"
fig.savefig(out_roc, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved: {out_roc}", flush=True)

# ── Confusion matrices ───────────────────────────────────────────────────────
print("\nGenerating confusion matrices...", flush=True)
fig2, axes2 = plt.subplots(1, 4, figsize=(15, 3.8))
fig2.subplots_adjust(wspace=0.42, left=0.05, right=0.98, top=0.85, bottom=0.14)
for col, ds_name in enumerate(DATASETS):
    ax = axes2[col]
    X, y, _ = fsets_all[ds_name]["combined"]
    print(f"    {ds_name} | combined", flush=True)
    cm = fold_cm(X, y)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"], fontsize=7.5, rotation=30, ha="right")
    ax.set_yticklabels(["Benign", "Attack"], fontsize=7.5)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.3f}\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("Predicted", fontsize=8)
    if col == 0:
        ax.set_ylabel("True", fontsize=8)
    ax.set_title(ds_name, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig2.suptitle("Confusion Matrices — Four Datasets, Combined Features, LightGBM (5-fold CV, Normalized by Row)",
              fontsize=10, fontweight="bold", y=0.98)
out_cm = FIGS / "cm_all_datasets.pdf"
fig2.savefig(out_cm, bbox_inches="tight", dpi=200)
plt.close(fig2)
print(f"  Saved: {out_cm}", flush=True)

# ── JSD empirical distribution ───────────────────────────────────────────────
print("\nGenerating JSD distribution figure...", flush=True)
fig3, axes3 = plt.subplots(1, 2, figsize=(7, 3.0))
for ax, (ds_name, jsd_hint) in zip(axes3, [("CICIDS-2017", "jsd_pkt_len"), ("UNSW-NB15", "jsd_pkt_sz")]):
    X, y, feat = fsets_all[ds_name]["combined"]
    jsd_col = jsd_hint if jsd_hint in feat else next(
        (f for f in feat if f.startswith("jsd_")), None)
    if jsd_col is None:
        ax.set_visible(False); continue
    idx = list(feat).index(jsd_col)
    jsd_vals = X[:, idx]
    rng = np.random.RandomState(RANDOM_STATE)
    benign_jsd = jsd_vals[y == 0]
    attack_jsd = jsd_vals[y == 1]
    if len(benign_jsd) > 5000:
        benign_jsd = rng.choice(benign_jsd, 5000, replace=False)
    if len(attack_jsd) > 5000:
        attack_jsd = rng.choice(attack_jsd, 5000, replace=False)
    x_grid = np.linspace(0, LN2 * 1.05, 300)
    for vals, color, label in [(benign_jsd, PRIMARY, "Benign"), (attack_jsd, ACCENT, "Attack")]:
        if len(np.unique(vals)) < 3:
            ax.hist(vals, bins=20, color=color, alpha=0.5, density=True, label=label)
        else:
            try:
                kde = gaussian_kde(vals, bw_method=0.15)
                ax.fill_between(x_grid, kde(x_grid), alpha=0.35, color=color)
                ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=label)
            except Exception:
                ax.hist(vals, bins=20, color=color, alpha=0.5, density=True, label=label)
    ax.axvline(LN2, color="black", ls="--", lw=1.0, label=r"$\ln 2$")
    ax.set_xlabel(f"{jsd_col} (nats)", fontsize=8)
    ax.set_ylabel("Density" if ax is axes3[0] else "", fontsize=8)
    ax.set_title(ds_name, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.tick_params(labelsize=7)
    ax.set_xlim(-0.01, LN2 * 1.08)
fig3.suptitle(r"Empirical JSD: attack flows concentrate near $\ln 2 \approx 0.693$ nats",
              fontsize=9, y=1.01)
plt.tight_layout()
out_jsd = FIGS / "jsd_empirical_dist.pdf"
fig3.savefig(out_jsd, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {out_jsd}", flush=True)

print("\nDone.", flush=True)
