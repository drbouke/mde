"""
Protocol checks that must pass before the experiments are run or their results are used.

  1. No identifier column and no label-like column is among the model features of any dataset.
  2. Every cross-directional JSD value lies in [0, ln 2] and no MDE feature is missing.
  3. The Gaussian JSD used by the pipeline agrees with adaptive integration
     (results/tables/jsd_validation.csv, written by validate_jsd.py).
  4. Every classifier of the comparison is trained on the full training fold and evaluated on the
     same folds (run_baselines.py and results/tables/baselines_comparison_folds.csv).
  5. Single features that nearly determine the label are listed, so that they are explained as
     dataset properties (results/tables/single_feature_auc.csv, written by jsd_stats.py).

Exit status is non-zero if any check fails.
"""
import re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from config import TABLES
from preprocess import load_dataset, clean
from fold_pipeline import MDEFeatures

META = ["binary_label", "multi_label", "label_name"]
LN2 = float(np.log(2.0))
IDENTIFIERS = {"flow id", "source ip", "src ip", "destination ip", "dst ip", "source port", "src port",
               "destination port", "dst port", "timestamp", "srcip", "dstip", "sport", "dsport",
               "stime", "ltime", "id"}
LABEL_LIKE = re.compile(r"label|attack_cat|difficulty|^class$|binary|multi", re.I)
TOL_TABLE, TOL_QUAD, AUC_FLAG = 1e-5, 1e-9, 0.95

failures, notes = [], []


def check(ok, message):
    print(("  ok    " if ok else "  FAIL  ") + message, flush=True)
    if not ok:
        failures.append(message)


print("1-2. feature matrices")
for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
    df = clean(load_dataset(ds))
    X = df.drop(columns=META, errors="ignore").select_dtypes(include=[np.number])
    ident = [c for c in X.columns if c.strip().lower() in IDENTIFIERS]
    lab = [c for c in X.columns if LABEL_LIKE.search(c)]
    check(not ident, f"{ds}: no identifier column among {X.shape[1]} features {ident if ident else ''}")
    check(not lab, f"{ds}: no label-like column among the features {lab if lab else ''}")
    t = MDEFeatures(ds, include_raw=False).fit(X)
    F = pd.DataFrame(t.transform(X), columns=t.mde_columns_)
    check(not F.isna().any().any(), f"{ds}: no missing value in the {F.shape[1]} MDE features")
    for c in [c for c in F.columns if "jsd" in c]:
        check(bool((F[c] >= 0).all() and (F[c] <= LN2 + 1e-12).all()), f"{ds}: {c} within [0, ln 2]")

print("3. numerical accuracy of the JSD")
p = TABLES / "jsd_validation.csv"
if p.exists():
    v = pd.read_csv(p)
    check(bool((v.table_max_err < TOL_TABLE).all()), f"tabulated JSD: maximum error {v.table_max_err.max():.1e} below {TOL_TABLE:.0e}")
    check(bool((v.quad_max_err < TOL_QUAD).all()), f"composite quadrature: maximum error {v.quad_max_err.max():.1e} below {TOL_QUAD:.0e}")
else:
    check(False, "jsd_validation.csv is missing; run pipeline/evaluation/validate_jsd.py")

print("4. classifier comparison")
src = (ROOT / "pipeline" / "experiments" / "run_baselines.py").read_text(encoding="utf-8")
check(bool(re.search(r"^SUBSAMPLE_N\s*=\s*None\b", src, re.M)), "run_baselines.py trains every model on the full training fold")
p = TABLES / "baselines_comparison_folds.csv"
if p.exists():
    f = pd.read_csv(p)
    per = f.groupby(["dataset", "fold"]).n.nunique()
    check(bool((per == 1).all()), "every model is evaluated on the same test folds")
    cnt = f.groupby("dataset").model.nunique()
    check(bool((cnt == cnt.max()).all()), f"every dataset has all {cnt.max()} models")
else:
    notes.append("baselines_comparison_folds.csv not found; comparison not checked")

print("5. single features that nearly determine the label")
p = TABLES / "single_feature_auc.csv"
if p.exists():
    a = pd.read_csv(p)
    for _, r in a[a.auc >= AUC_FLAG].iterrows():
        notes.append(f"{r.dataset}: {r.feature} alone reaches AUC {r.auc:.3f}; explain it as a dataset property")
else:
    notes.append("single_feature_auc.csv not found; run pipeline/evaluation/jsd_stats.py")

for n in notes:
    print("  note  " + n)
print(f"\n{len(failures)} failed check(s)")
sys.exit(1 if failures else 0)
