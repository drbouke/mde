"""
Descriptive statistics of the features, computed on each full sample.

  jsd_empirical_stats.csv   per dataset and class, and per attack family: share of flows at the
                            bound ln 2, share of nearly symmetric flows, median and mean of each
                            cross-directional JSD feature
  mde_score_delta.csv       mean composite MDE score of attack and benign flows and their difference
  single_feature_auc.csv    ROC-AUC of every single feature (raw and MDE) taken as a score by itself,
                            reported as max(AUC, 1 - AUC)

Output: results/tables/
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import TABLES
from preprocess import load_dataset, clean
from fold_pipeline import MDEFeatures

META = ["binary_label", "multi_label", "label_name"]
LN2 = float(np.log(2.0))
AT_BOUND = LN2 - 1e-3        # within 0.001 nats of ln 2
SYMMETRIC = 0.05


def stats(v):
    return {"n": int(len(v)), "share_at_bound": float((v >= AT_BOUND).mean()),
            "share_below_0.05": float((v < SYMMETRIC).mean()), "median": float(np.median(v)),
            "mean": float(np.mean(v))}


rows, deltas, aucs = [], [], []
for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
    df = clean(load_dataset(ds))
    X = df.drop(columns=META, errors="ignore").select_dtypes(include=[np.number])
    y = df["binary_label"].values
    fam = df["label_name"].astype(str).values if "label_name" in df.columns else None
    t = MDEFeatures(ds, include_raw=False).fit(X)
    F = pd.DataFrame(t.transform(X), columns=t.mde_columns_)

    sc = F["mde_score"].to_numpy()
    deltas.append({"dataset": ds, "attack_mean": float(sc[y == 1].mean()), "benign_mean": float(sc[y == 0].mean()),
                   "delta": float(sc[y == 1].mean() - sc[y == 0].mean())})

    both = pd.concat([X.reset_index(drop=True), F], axis=1)
    for c in both.columns:
        v = both[c].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if fin.sum() < 2 or np.unique(v[fin]).size < 2:
            continue
        v = np.where(fin, v, np.nanmedian(v[fin]))
        a = roc_auc_score(y, v)
        aucs.append({"dataset": ds, "feature": c, "is_mde": c in F.columns, "auc": float(max(a, 1.0 - a))})

    for c in [c for c in F.columns if "jsd" in c]:
        v = F[c].to_numpy()
        rows.append({"dataset": ds, "feature": c, "group": "benign", **stats(v[y == 0])})
        rows.append({"dataset": ds, "feature": c, "group": "attack", **stats(v[y == 1])})
        if fam is not None:
            for name in sorted(set(fam[y == 1])):
                m = (fam == name) & (y == 1)
                if m.sum() >= 50:
                    rows.append({"dataset": ds, "feature": c, "group": f"family:{name}", **stats(v[m])})

TABLES.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(TABLES / "jsd_empirical_stats.csv", index=False)
dl = pd.DataFrame(deltas)
dl.to_csv(TABLES / "mde_score_delta.csv", index=False)
au = pd.DataFrame(aucs).sort_values(["dataset", "auc"], ascending=[True, False])
au.to_csv(TABLES / "single_feature_auc.csv", index=False)

pd.set_option("display.width", 200)
print(dl.round(4).to_string(index=False))
print(au.groupby("dataset").head(4).round(4).to_string(index=False))
print(au[au.is_mde].groupby("dataset").head(3).round(4).to_string(index=False))
print("Saved: jsd_empirical_stats.csv, mde_score_delta.csv, single_feature_auc.csv in", TABLES)
