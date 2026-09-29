"""
Unseen-attack-family evaluation on CICIDS-2017.

Train on all attack families except Infiltration and Bot;
test on Infiltration + Bot + matched benign.

Outputs:
  results/tables/unseen_attack_eval.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
)
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import make_lgb_pipeline, make_rf_pipeline
from metrics import full_metrics

TABLES.mkdir(parents=True, exist_ok=True)
HOLDOUT_FAMILIES = ["Infiltration", "Bot"]


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


def align(X_tr, feat_tr, X_te, feat_te):
    feat_tr, feat_te = list(feat_tr), list(feat_te)
    common = [f for f in feat_tr if f in feat_te]
    return (X_tr[:, [feat_tr.index(f) for f in common]],
            X_te[:, [feat_te.index(f) for f in common]],
            common)


raw = load_dataset("CICIDS-2017")
print(f"\nAttack families: {raw['label_name'].value_counts().to_dict()}", flush=True)

train_mask = ~raw["label_name"].isin(HOLDOUT_FAMILIES)
test_mask  = raw["label_name"].isin(HOLDOUT_FAMILIES) | (raw["label_name"] == "BENIGN")
df_train_raw = raw[train_mask].reset_index(drop=True)
df_test_raw  = raw[test_mask].reset_index(drop=True)

print(f"Train: {len(df_train_raw):,} flows", flush=True)
print(f"Test:  {len(df_test_raw):,} flows  "
      f"(class dist: {df_test_raw['binary_label'].value_counts().to_dict()})", flush=True)

df_tr = clean(df_train_raw)
df_te = clean(df_test_raw)
mde_input_tr, mde_stats = clean_for_mde(df_train_raw, return_stats=True)
mde_input_te = clean_for_mde(df_test_raw, stats=mde_stats)
mde_tr = compute_mde(mde_input_tr, "CICIDS-2017")
mde_te = compute_mde(mde_input_te, "CICIDS-2017", fit_df=mde_input_tr)
fsets_tr = build_feature_sets(df_tr, mde_tr)
fsets_te = build_feature_sets(df_te, mde_te)
y_tr = fsets_tr["conventional"][1]
y_te = fsets_te["conventional"][1]

rows = []
for ablation in ["conventional", "entropy_only", "combined"]:
    X_tr_raw, _, feat_tr = fsets_tr[ablation]
    X_te_raw, _, feat_te = fsets_te[ablation]
    X_tr_a, X_te_a, _ = align(X_tr_raw, feat_tr, X_te_raw, feat_te)

    for model_name, clf_fn in [("LightGBM", make_lgb), ("RandomForest", make_rf)]:
        pipe = make_lgb_pipeline(clf_fn()) if model_name == "LightGBM" \
            else make_rf_pipeline(clf_fn())
        pipe.fit(X_tr_a, y_tr)
        y_pred = pipe.predict(X_te_a)
        y_prob = pipe.predict_proba(X_te_a)[:, 1]

        m = full_metrics(y_te, y_pred, y_prob)
        print(f"  [{ablation:14s}] {model_name:12s} | "
              f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}  DR={m['dr']:.4f}  FAR={m['far']:.4f}", flush=True)
        rows.append({
            "holdout_families": str(HOLDOUT_FAMILIES),
            "model": model_name, "ablation": ablation,
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "n_feat": X_tr_a.shape[1], **m,
            "dr_attack": m["dr"], "far_benign": m["far"],
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(TABLES / "unseen_attack_eval.csv", index=False)
print(f"\nSaved: {TABLES / 'unseen_attack_eval.csv'}", flush=True)
