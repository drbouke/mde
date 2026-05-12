"""
Pseudo-live temporal replay evaluation: CICIDS-2017 Mon–Thu → Friday.

Trains the complete pipeline on Mon–Thu traffic, freezes all parameters,
then replays Friday traffic chronologically as a simulated live stream with
no labels available during inference and no retraining during replay.
Operational metrics are reported over 20 equal-sized chronological windows
and compared under two threshold strategies: fixed (0.5) versus Youden's-J
threshold derived from training-set probabilities.

Outputs:
  results/tables/temporal_replay_windows.csv   — per-window metrics
  results/tables/temporal_replay_summary.csv   — overall summary
  results/figures/temporal_replay_drift.pdf    — DR, FAR, MCC, AUC over time
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from config import DATA, TABLES, FIGS, RANDOM_STATE
from preprocess import clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

AUROC_THRESHOLD = 0.99
N_WINDOWS       = 20
SAMPLE_TRAIN    = 150_000

CIC17_DIR   = DATA / "CICIDS2017" / "TrafficLabelling_"
FRI_FILES   = sorted(f for f in CIC17_DIR.glob("*.csv") if "Friday"  in f.name)
TRAIN_FILES = sorted(f for f in CIC17_DIR.glob("*.csv") if "Friday" not in f.name)

# Identifier columns to drop before feature construction
DROP_COLS    = ["Flow ID", "Source IP", "Source Port",
                "Destination IP", "Destination Port"]
TIMESTAMP_COL = "Timestamp"


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_frame(df):
    """Strip columns, deduplicate, assign labels, drop identifiers."""
    df.columns = df.columns.str.strip()
    seen, new_cols = {}, []
    for c in df.columns:
        seen[c] = seen.get(c, -1) + 1
        new_cols.append(f"{c}_{seen[c]}" if seen[c] else c)
    df.columns = new_cols
    lc = "Label"
    df[lc] = df[lc].astype(str).str.strip()
    df["binary_label"] = (~df[lc].isin(["BENIGN"])).astype(int)
    df["multi_label"]  = LabelEncoder().fit_transform(df[lc])
    df["label_name"]   = df[lc]
    for c in DROP_COLS + [lc]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def load_train_files(file_list, sample_n):
    frames = []
    for fp in file_list:
        try:
            df = pd.read_csv(fp, encoding="latin-1", low_memory=False)
            df = _parse_frame(df)
            if TIMESTAMP_COL in df.columns:
                df.drop(columns=[TIMESTAMP_COL], inplace=True)
            frames.append(df)
        except Exception as e:
            print(f"  Skip {fp.name}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > sample_n:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_n,
                                     random_state=RANDOM_STATE)
        idx, _ = next(sss.split(combined, combined["binary_label"]))
        combined = combined.iloc[idx].reset_index(drop=True)
    return combined


def load_friday_ordered(file_list):
    """Load Friday CSVs and sort chronologically by Timestamp."""
    frames = []
    has_ts = False
    for fp in file_list:
        try:
            df = pd.read_csv(fp, encoding="latin-1", low_memory=False)
            df = _parse_frame(df)
            if TIMESTAMP_COL in df.columns:
                has_ts = True
            frames.append(df)
        except Exception as e:
            print(f"  Skip {fp.name}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    if has_ts and TIMESTAMP_COL in combined.columns:
        combined[TIMESTAMP_COL] = pd.to_datetime(
            combined[TIMESTAMP_COL], dayfirst=True, errors="coerce"
        )
        combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        combined.drop(columns=[TIMESTAMP_COL], inplace=True)
        print(f"  Sorted {len(combined):,} Friday flows chronologically")
    else:
        print(f"  No Timestamp found; using file order ({len(combined):,} flows)")
    return combined


# ── Helpers ───────────────────────────────────────────────────────────────────

def align(X_tr, feat_tr, X_te, feat_te):
    feat_tr, feat_te = list(feat_tr), list(feat_te)
    common = [f for f in feat_tr if f in feat_te]
    return (X_tr[:, [feat_tr.index(f) for f in common]],
            X_te[:, [feat_te.index(f) for f in common]],
            common)


def debias_mask(X_tr, y_tr, feat_names, thr=AUROC_THRESHOLD):
    keep, removed = [], []
    for i, name in enumerate(feat_names):
        try:
            auc = roc_auc_score(y_tr, X_tr[:, i])
            auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
        (removed if auc >= thr else keep).append((i, name, round(auc, 4)))
    for _, name, auc in removed:
        print(f"    removed: {name:45s} AUC={auc}")
    print(f"    Kept {len(keep)}/{len(feat_names)} features after debiasing")
    return [i for i, *_ in keep]


def youden_threshold(y_tr, y_prob_tr):
    """Optimal decision threshold from Youden's J statistic on training scores."""
    fpr, tpr, thresholds = roc_curve(y_tr, y_prob_tr)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def window_metrics(y_true, y_pred, y_prob):
    """Per-window operational metrics; returns None if only one class present."""
    if len(np.unique(y_true)) < 2:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    m = {
        "dr":  round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "far": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "f1":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    try:
        m["auc"]   = round(roc_auc_score(y_true, y_prob), 4)
        m["prauc"] = round(average_precision_score(y_true, y_prob, pos_label=1), 4)
    except Exception:
        m["auc"] = m["prauc"] = float("nan")
    return m


# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading Mon–Thu (train)...", flush=True)
df_tr_raw = load_train_files(TRAIN_FILES, SAMPLE_TRAIN)
print(f"  {len(df_tr_raw):,} rows | Attack: {df_tr_raw['binary_label'].sum():,}", flush=True)

print("\nLoading Friday (test — chronological)...", flush=True)
df_te_raw = load_friday_ordered(FRI_FILES)
print(f"  {len(df_te_raw):,} rows | Attack: {df_te_raw['binary_label'].sum():,}", flush=True)

# ── Feature construction ──────────────────────────────────────────────────────
df_tr = clean(df_tr_raw)
df_te = clean(df_te_raw)
mde_tr = compute_mde(clean_for_mde(df_tr_raw), "CICIDS-2017")
mde_te = compute_mde(clean_for_mde(df_te_raw), "CICIDS-2017")
fsets_tr = build_feature_sets(df_tr, mde_tr)
fsets_te = build_feature_sets(df_te, mde_te)
y_tr = fsets_tr["conventional"][1]
y_te = fsets_te["conventional"][1]

# ── Debiasing — same protocol as run_timesplit.py ─────────────────────────────
print("\nDebiasing conventional features...", flush=True)
X_tr_conv, X_te_conv, feat_conv = align(
    fsets_tr["conventional"][0], fsets_tr["conventional"][2],
    fsets_te["conventional"][0], fsets_te["conventional"][2])
keep_conv = debias_mask(X_tr_conv, y_tr, feat_conv)
X_tr_conv, X_te_conv = X_tr_conv[:, keep_conv], X_te_conv[:, keep_conv]

print("\nDebiasing entropy-only features...", flush=True)
X_tr_ent, X_te_ent, feat_ent = align(
    fsets_tr["entropy_only"][0], fsets_tr["entropy_only"][2],
    fsets_te["entropy_only"][0], fsets_te["entropy_only"][2])
keep_ent = debias_mask(X_tr_ent, y_tr, feat_ent)
X_tr_ent, X_te_ent = X_tr_ent[:, keep_ent], X_te_ent[:, keep_ent]

X_tr_comb = np.concatenate([X_tr_conv, X_tr_ent], axis=1)
X_te_comb = np.concatenate([X_te_conv, X_te_ent], axis=1)

print(f"\nDebiased feature counts: conv={X_tr_conv.shape[1]}, "
      f"ent={X_tr_ent.shape[1]}, combined={X_tr_comb.shape[1]}", flush=True)

# ── Replay loop ───────────────────────────────────────────────────────────────
CONDITIONS = [
    ("conventional", X_tr_conv, X_te_conv),
    ("entropy_only", X_tr_ent,  X_te_ent),
    ("combined",     X_tr_comb, X_te_comb),
]

window_rows  = []
summary_rows = []

for ablation, X_tr_use, X_te_use in CONDITIONS:
    print(f"\n[{ablation}]", flush=True)

    # Fit and freeze preprocessing on Mon–Thu
    imp = SimpleImputer(strategy="median").fit(X_tr_use)
    cli = PercentileClipper().fit(imp.transform(X_tr_use))
    Xtr = cli.transform(imp.transform(X_tr_use))
    Xte = cli.transform(imp.transform(X_te_use))

    # Train classifier — parameters frozen after this point
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )
    clf.fit(Xtr, y_tr)

    # Derive Youden threshold from training-set probabilities
    y_prob_tr  = clf.predict_proba(Xtr)[:, 1]
    thr_fixed  = 0.5
    thr_youden = youden_threshold(y_tr, y_prob_tr)
    print(f"  Fixed threshold: {thr_fixed:.3f}  |  "
          f"Youden threshold: {thr_youden:.4f}", flush=True)

    # Chronological replay: divide Friday into N_WINDOWS equal slices
    n_test      = len(y_te)
    window_size = n_test // N_WINDOWS

    for w in range(N_WINDOWS):
        start = w * window_size
        end   = (w + 1) * window_size if w < N_WINDOWS - 1 else n_test
        Xw, yw = Xte[start:end], y_te[start:end]

        # Inference — no labels used, no retraining
        t0 = time.perf_counter()
        y_prob_w = clf.predict_proba(Xw)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000

        y_pred_fixed  = (y_prob_w >= thr_fixed).astype(int)
        y_pred_youden = (y_prob_w >= thr_youden).astype(int)

        m_fixed  = window_metrics(yw, y_pred_fixed,  y_prob_w)
        m_youden = window_metrics(yw, y_pred_youden, y_prob_w)
        if m_fixed is None:
            continue

        row = {
            "ablation":    ablation,
            "window":      w + 1,
            "flow_start":  start,
            "flow_end":    end,
            "n_flows":     end - start,
            "n_attack":    int(yw.sum()),
            "latency_ms":  round(latency_ms, 3),
        }
        for k, v in m_fixed.items():
            row[f"fixed_{k}"] = v
        for k, v in m_youden.items():
            row[f"youden_{k}"] = v
        window_rows.append(row)

        print(f"  W{w+1:02d}/{N_WINDOWS} n={end-start:5d} atk={int(yw.sum()):4d} | "
              f"DR fixed={m_fixed['dr']:.3f} youden={m_youden['dr']:.3f}  "
              f"FAR fixed={m_fixed['far']:.3f}  "
              f"AUC={m_fixed.get('auc', float('nan')):.3f}  "
              f"lat={latency_ms:.1f}ms", flush=True)

    # Overall summary across all Friday flows
    y_prob_all        = clf.predict_proba(Xte)[:, 1]
    y_pred_all_fixed  = (y_prob_all >= thr_fixed).astype(int)
    y_pred_all_youden = (y_prob_all >= thr_youden).astype(int)
    m_all_f = window_metrics(y_te, y_pred_all_fixed,  y_prob_all)
    m_all_y = window_metrics(y_te, y_pred_all_youden, y_prob_all)

    sr = {"ablation": ablation, "n_train": len(y_tr), "n_test": n_test,
          "n_windows": N_WINDOWS, "thr_fixed": thr_fixed,
          "thr_youden": round(thr_youden, 4)}
    for k, v in m_all_f.items():
        sr[f"fixed_{k}"] = v
    for k, v in m_all_y.items():
        sr[f"youden_{k}"] = v
    summary_rows.append(sr)

    print(f"  Overall | DR fixed={m_all_f['dr']:.4f} youden={m_all_y['dr']:.4f}  "
          f"AUC={m_all_f.get('auc', float('nan')):.4f}", flush=True)

# ── Save tables ───────────────────────────────────────────────────────────────
df_windows = pd.DataFrame(window_rows)
df_summary = pd.DataFrame(summary_rows)
df_windows.to_csv(TABLES / "temporal_replay_windows.csv", index=False)
df_summary.to_csv(TABLES / "temporal_replay_summary.csv", index=False)
print(f"\nSaved: {TABLES / 'temporal_replay_windows.csv'}", flush=True)
print(f"Saved: {TABLES / 'temporal_replay_summary.csv'}", flush=True)

# ── Temporal drift figure ─────────────────────────────────────────────────────
print("\nGenerating temporal drift figure...", flush=True)

COND_COLORS = {"conventional": "#4878d0", "entropy_only": "#ee854a", "combined": "#6acc65"}
METRICS_PLOT = [
    ("dr",    "Detection Rate (DR)"),
    ("far",   "False Alarm Rate (FAR)"),
    ("mcc",   "MCC"),
    ("auc",   "ROC-AUC"),
]

fig, axes = plt.subplots(len(METRICS_PLOT), 1,
                          figsize=(9, 2.6 * len(METRICS_PLOT)), sharex=True)
fig.subplots_adjust(hspace=0.32, left=0.09, right=0.97, top=0.93, bottom=0.06)

for ax, (metric, label) in zip(axes, METRICS_PLOT):
    for cond, color in COND_COLORS.items():
        sub = df_windows[df_windows["ablation"] == cond].sort_values("window")
        if sub.empty:
            continue
        fc = f"fixed_{metric}"
        yc = f"youden_{metric}"
        if fc in sub.columns:
            ax.plot(sub["window"], sub[fc], color=color, lw=1.8,
                    marker="o", ms=4, label=cond.replace("_", " ").title())
        if yc in sub.columns:
            ax.plot(sub["window"], sub[yc], color=color, lw=1.2,
                    ls="--", marker="s", ms=3, alpha=0.65)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    if ax is axes[0]:
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, fontsize=7, loc="upper right",
                  framealpha=0.85, title="Solid=fixed thr  Dashed=Youden thr",
                  title_fontsize=6)

axes[-1].set_xlabel("Replay Window (chronological order)", fontsize=9)
fig.suptitle(
    "Pseudo-Live Temporal Replay — CICIDS-2017 Mon–Thu → Friday\n"
    "Operational Metrics over 20 Chronological Windows "
    "(Fixed vs Youden Threshold)",
    fontsize=10, fontweight="bold",
)

out_fig = FIGS / "temporal_replay_drift.pdf"
fig.savefig(out_fig, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out_fig}", flush=True)
print("\nDone.", flush=True)
