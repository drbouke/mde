"""
Mean |SHAP| ranking of every feature per dataset (LightGBM, combined features, fitted on the full
sample as in the SHAP figures) and the top contributors of the most confidently classified attack
and benign instances, so that the prose about SHAP rankings can be checked against numbers.

Output: results/tables/shap_ranks.csv and results/tables/shap_top_instances.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean
from fold_pipeline import MDEFeatures

META = ["binary_label", "multi_label", "label_name"]
MDE_PREFIXES = ("jsd_", "ade_", "dir_entropy", "flag_entropy", "log_", "mde_score", "conn_state_entropy",
                "srv_diversity_entropy", "byte_asym_jsd", "ttl_asym_entropy")


def is_mde(name):
    return any(name.startswith(p) for p in MDE_PREFIXES)


rank_rows, inst_rows = [], []
for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
    raw = load_dataset(ds); df = clean(raw)
    X_raw = df.drop(columns=META, errors="ignore").select_dtypes(include=[np.number]); y = df["binary_label"].values
    t = MDEFeatures(ds, include_raw=True).fit(X_raw)
    X = t.transform(X_raw); feat = list(t.feature_names_out_)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight="balanced",
                             n_jobs=-1, random_state=RANDOM_STATE, verbose=-1).fit(X, y)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X.shape[0], min(600, X.shape[0]), replace=False)
    X_sub = X[idx]
    exp = shap.TreeExplainer(clf)(X_sub)
    sv = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    for r, i in enumerate(order, start=1):
        rank_rows.append({"dataset": ds, "rank": r, "feature": feat[i], "is_mde": is_mde(feat[i]), "mean_abs_shap": round(float(imp[i]), 5)})
    proba = clf.predict_proba(X_sub)[:, 1]
    for tag, pos in (("attack", int(np.argmax(proba))), ("benign", int(np.argmin(proba)))):
        contrib = sv[pos]
        top = np.argsort(-np.abs(contrib))[:5]
        for r, i in enumerate(top, start=1):
            inst_rows.append({"dataset": ds, "instance": tag, "prob_attack": round(float(proba[pos]), 4), "rank": r,
                              "feature": feat[i], "value": round(float(X_sub[pos, i]), 4), "shap": round(float(contrib[i]), 4)})
    print(ds, "top-5:", [feat[i] for i in order[:5]], flush=True)
pd.DataFrame(rank_rows).to_csv(TABLES / "shap_ranks.csv", index=False)
pd.DataFrame(inst_rows).to_csv(TABLES / "shap_top_instances.csv", index=False)
print("saved shap_ranks.csv, shap_top_instances.csv")
