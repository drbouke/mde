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
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import make_lgb_pipeline, make_rf_pipeline

TABLES.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
DATASETS = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]


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


def run_cv(X, y, model_name, ablation, ds_name):
    clf_base = make_lgb() if model_name == "LightGBM" else make_rf()
    pipe = make_lgb_pipeline(clf_base) if model_name == "LightGBM" \
        else make_rf_pipeline(clf_base)
    s = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=1)
    row = {
        "dataset":  ds_name,
        "model":    model_name,
        "ablation": ablation,
        "n_feat":   X.shape[1],
        "f1":       round(s["test_f1_weighted"].mean(),        4),
        "f1_std":   round(s["test_f1_weighted"].std(),         4),
        "prec":     round(s["test_precision_weighted"].mean(), 4),
        "rec":      round(s["test_recall_weighted"].mean(),    4),
        "dr":       round(s["test_dr"].mean(),                 4),
        "far":      round(s["test_far"].mean(),                4),
        "acc":      round(s["test_accuracy"].mean(),           4),
        "mcc":      round(s["test_mcc"].mean(),                4),
        "auc":      round(s["test_roc_auc"].mean(),            4),
        "prauc":    round(s["test_prauc"].mean(),              4),
    }
    print(
        f"  [{ds_name}] {model_name:12s} | {ablation:14s} | "
        f"F1={row['f1']:.4f} (±{row['f1_std']:.4f})  "
        f"DR={row['dr']:.4f}  FAR={row['far']:.4f}  MCC={row['mcc']:.4f}",
        flush=True,
    )
    return row


rows = []
for ds_name in DATASETS:
    print(f"\n{'='*65}", flush=True)
    print(f"  DATASET: {ds_name}", flush=True)
    print(f"{'='*65}", flush=True)
    raw          = load_dataset(ds_name)
    df_clean     = clean(raw)
    df_mde_input = clean_for_mde(raw)
    mde          = compute_mde(df_mde_input, ds_name)
    fsets        = build_feature_sets(df_clean, mde)
    y_ref = fsets["conventional"][1]
    print(f"  y: {np.bincount(y_ref)} (benign / attack)", flush=True)
    for ablation in ["conventional", "entropy_only", "combined"]:
        X, y, _ = fsets[ablation]
        for model_name in ["LightGBM", "RandomForest"]:
            rows.append(run_cv(X, y, model_name, ablation, ds_name))

df_out = pd.DataFrame(rows)
df_out.to_csv(TABLES / "ablation_cv.csv", index=False)
print(f"\nSaved: {TABLES / 'ablation_cv.csv'}", flush=True)
