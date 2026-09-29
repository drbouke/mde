"""
Main ablation: 5-fold stratified CV on 4 datasets × 3 feature conditions × 2 classifiers.

Outputs:
  results/tables/ablation_cv.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, matthews_corrcoef, average_precision_score,
    recall_score, confusion_matrix,
)
import lightgbm as lgb

from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean
from fold_pipeline import make_lgb_pipeline, make_rf_pipeline, MDEFeatures
from metrics import cv_full_metrics
from sklearn.base import clone

TABLES.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
DATASETS = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]
META = ["binary_label", "multi_label", "label_name"]


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )


def _dr(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def _far(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0


def _prauc(estimator, X, y):
    return average_precision_score(y, estimator.predict_proba(X)[:, 1], pos_label=1)


SCORING = {
    "f1_weighted":        "f1_weighted",
    "roc_auc":            "roc_auc",
    "accuracy":           "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted":    "recall_weighted",
    "dr":                 make_scorer(_dr),
    "far":                make_scorer(_far),
    "mcc":                make_scorer(matthews_corrcoef),
    "prauc":              _prauc,
}


def run_cv(X, y, model_name, ablation, ds_name, mde=None):
    """5-fold CV with the complete metric suite; when `mde` is an MDEFeatures transformer, X is
    the raw-statistics frame and the entropy features are built inside each fold."""
    n_feat = {"v": X.shape[1]}

    def fit_predict(tr, te):
        clf_base = make_lgb() if model_name == "LightGBM" else make_rf()
        pipe = make_lgb_pipeline(clf_base, mde=clone(mde) if mde is not None else None) if model_name == "LightGBM" \
            else make_rf_pipeline(clf_base, mde=clone(mde) if mde is not None else None)
        Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        Xte = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
        pipe.fit(Xtr, y[tr])
        if mde is not None:
            n_feat["v"] = len(pipe.named_steps["mde"].feature_names_out_)
        return pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]

    summary, folds = cv_full_metrics(fit_predict, X, y, CV, label=f"{ds_name} {model_name} {ablation}")
    folds.insert(0, "ablation", ablation); folds.insert(0, "model", model_name); folds.insert(0, "dataset", ds_name)
    fold_rows.append(folds)
    row = {"dataset": ds_name, "model": model_name, "ablation": ablation, "n_feat": n_feat["v"], **summary}
    print(
        f"  [{ds_name}] {model_name:12s} | {ablation:14s} | "
        f"F1={row['f1']:.4f} (±{row['f1_std']:.4f})  "
        f"DR={row['dr']:.4f}  FAR={row['far']:.4f}  MCC={row['mcc']:.4f}",
        flush=True,
    )
    return row


rows = []
fold_rows = []
for ds_name in DATASETS:
    print(f"\n{'='*65}", flush=True)
    print(f"  DATASET: {ds_name}", flush=True)
    print(f"{'='*65}", flush=True)
    raw      = load_dataset(ds_name)
    df_clean = clean(raw)
    X_raw    = df_clean.drop(columns=META, errors="ignore").select_dtypes(include=[np.number])
    y        = df_clean["binary_label"].values
    print(f"  y: {np.bincount(y)} (benign / attack)", flush=True)
    for ablation in ["conventional", "entropy_only", "combined"]:
        for model_name in ["LightGBM", "RandomForest"]:
            mde = None if ablation == "conventional" else MDEFeatures(ds_name, include_raw=(ablation == "combined"))
            X = X_raw.values if mde is None else X_raw
            rows.append(run_cv(X, y, model_name, ablation, ds_name, mde))

df_out = pd.DataFrame(rows)
df_out.to_csv(TABLES / "ablation_cv.csv", index=False)
pd.concat(fold_rows, ignore_index=True).to_csv(TABLES / "ablation_cv_folds.csv", index=False)
print(f"\nSaved: {TABLES / 'ablation_cv.csv'}", flush=True)
