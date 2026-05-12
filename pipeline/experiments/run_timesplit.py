"""
CICIDS-2017 temporal generalization: Mon–Thu → Friday.

Removes artefact features (single-feature AUROC > 0.99 on train) before evaluation.

Outputs:
  results/tables/cicids2017_timesplit.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from config import DATA, TABLES, RANDOM_STATE
from preprocess import clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper

TABLES.mkdir(parents=True, exist_ok=True)
AUROC_THRESHOLD = 0.99

CIC17_DIR   = DATA / "CICIDS2017" / "TrafficLabelling_"
FRI_FILES   = sorted(f for f in CIC17_DIR.glob("*.csv") if "Friday"  in f.name)
TRAIN_FILES = sorted(f for f in CIC17_DIR.glob("*.csv") if "Friday" not in f.name)
SAMPLE_TRAIN, SAMPLE_TEST = 150_000, 100_000
DROP_COLS = ["Flow ID", "Source IP", "Source Port",
             "Destination IP", "Destination Port", "Timestamp"]


def load_files(file_list, max_per_file=None):
    """Load CSV files, process labels and drop columns per-file to bound memory."""
    frames = []
    for fp in file_list:
        try:
            df = pd.read_csv(fp, encoding="latin-1", low_memory=False)
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
            # Drop identifier and label columns per-file before concat
            for c in DROP_COLS + [lc]:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            if max_per_file and len(df) > max_per_file:
                df = df.sample(n=max_per_file, random_state=RANDOM_STATE)
            frames.append(df)
        except Exception as e:
            print(f"  Skip {fp.name}: {e}")
    return pd.concat(frames, ignore_index=True)


def strat_sample(df, n):
    if len(df) <= n:
        return df
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=RANDOM_STATE)
    idx, _ = next(sss.split(df, df["binary_label"]))
    return df.iloc[idx].reset_index(drop=True)


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


def full_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {
        "f1":    round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "prec":  round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "rec":   round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "dr":    round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "far":   round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "acc":   round(accuracy_score(y_true, y_pred), 4),
        "mcc":   round(matthews_corrcoef(y_true, y_pred), 4),
        "auc":   round(roc_auc_score(y_true, y_prob), 4),
        "prauc": round(average_precision_score(y_true, y_prob, pos_label=1), 4),
    }


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
    )


def align(X_tr, feat_tr, X_te, feat_te):
    feat_tr, feat_te = list(feat_tr), list(feat_te)
    common = [f for f in feat_tr if f in feat_te]
    return (X_tr[:, [feat_tr.index(f) for f in common]],
            X_te[:, [feat_te.index(f) for f in common]],
            common)


print("\nLoading Mon–Thu (train)...", flush=True)
df_tr_raw = strat_sample(load_files(TRAIN_FILES), SAMPLE_TRAIN)
print(f"  {len(df_tr_raw):,} rows", flush=True)

print("\nLoading Friday (test)...", flush=True)
df_te_raw = strat_sample(load_files(FRI_FILES), SAMPLE_TEST)
print(f"  {len(df_te_raw):,} rows", flush=True)

df_tr = clean(df_tr_raw)
df_te = clean(df_te_raw)
mde_tr = compute_mde(clean_for_mde(df_tr_raw), "CICIDS-2017")
mde_te = compute_mde(clean_for_mde(df_te_raw), "CICIDS-2017")
fsets_tr = build_feature_sets(df_tr, mde_tr)
fsets_te = build_feature_sets(df_te, mde_te)
y_tr = fsets_tr["conventional"][1]
y_te = fsets_te["conventional"][1]

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

print(f"\nDebiased: conv={X_tr_conv.shape[1]}, ent={X_tr_ent.shape[1]}, "
      f"comb={X_tr_comb.shape[1]}", flush=True)

rows = []
for model_name, clf in [("LightGBM", make_lgb()), ("RandomForest", make_rf())]:
    for ablation, X_tr_use, X_te_use in [
        ("conventional", X_tr_conv, X_te_conv),
        ("entropy_only", X_tr_ent,  X_te_ent),
        ("combined",     X_tr_comb, X_te_comb),
    ]:
        imp = SimpleImputer(strategy="median").fit(X_tr_use)
        cli = PercentileClipper().fit(imp.transform(X_tr_use))
        Xtr = cli.transform(imp.transform(X_tr_use))
        Xte = cli.transform(imp.transform(X_te_use))
        clf.fit(Xtr, y_tr)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]
        m = full_metrics(y_te, y_pred, y_prob)
        print(f"  [{model_name:12s}] {ablation:14s} n={X_tr_use.shape[1]} | "
              f"F1={m['f1']:.4f}  DR={m['dr']:.4f}  "
              f"FAR={m['far']:.4f}  AUC={m['auc']:.4f}", flush=True)
        rows.append({
            "model": model_name, "ablation": ablation,
            "n_train": len(y_tr), "n_test": len(y_te),
            "n_feat": X_tr_use.shape[1], **m,
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(TABLES / "cicids2017_timesplit.csv", index=False)
print(f"\nSaved: {TABLES / 'cicids2017_timesplit.csv'}", flush=True)
