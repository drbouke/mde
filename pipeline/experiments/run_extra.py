"""
Cross-dataset transfer, noise robustness, entropy-versus-simple-statistics, and computational
profiling. Entropy features are built from training-portion statistics in every split.

Outputs (results/tables): transfer.csv, robustness.csv, simplestats.csv, profiling.csv
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, NON_ENTROPY_DESCRIPTORS
from fold_pipeline import MDEFeatures, make_lgb_pipeline, make_rf_pipeline
from metrics import full_metrics, aggregate

META = ["binary_label", "multi_label", "label_name"]
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None


def wanted(section):
    return ONLY is None or ONLY == section
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
TABLES.mkdir(parents=True, exist_ok=True)

SIMPLE_STATS = {
    "NSL-KDD": ["src_bytes", "dst_bytes", "count", "srv_count", "serror_rate", "srv_serror_rate",
                "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"],
    "CICIDS-2017": ["Fwd Packet Length Mean", "Fwd Packet Length Std", "Fwd Packet Length Max", "Fwd Packet Length Min",
                    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
                    "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std"],
    "CICIDS-2018": ["Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
                    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
                    "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std"],
    "UNSW-NB15": ["sbytes", "dbytes", "Spkts", "Dpkts", "smeansz", "dmeansz", "Sjit", "Djit", "Sintpkt", "Dintpkt"],
}


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)


def load_xy(ds):
    raw = load_dataset(ds)
    df = clean(raw)
    X = df.drop(columns=META, errors="ignore").select_dtypes(include=[np.number])
    return raw, X, df["binary_label"].values


data = {ds: load_xy(ds) for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]}

# ── Cross-dataset transfer (entropy-only, LightGBM) ────────────────────────────────────────
if wanted("transfer"):
    print("\n=== Transfer ===", flush=True)
    rows = []
    for src, tgt in [("CICIDS-2017", "CICIDS-2018"), ("CICIDS-2018", "CICIDS-2017")]:
        raw_s, Xs, ys = data[src]
        raw_t, Xt, yt = data[tgt]
        tr, te = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=RANDOM_STATE)
        t = MDEFeatures(src, include_raw=False).fit(Xs.iloc[tr])
        pipe = make_lgb_pipeline(make_lgb()).fit(t.transform(Xs.iloc[tr]), ys[tr])
        p = pipe.predict_proba(t.transform(Xs.iloc[te]))[:, 1]
        rows.append({"source": src, "target": src, "kind": "hold-out", "n_train": len(tr), "n_test": len(te),
                     **full_metrics(ys[te], (p >= 0.5).astype(int), p)})
        # zero-shot: model and feature statistics from the full source; target inputs imputed with their
        # own medians (no label use), entropy columns aligned by name, score bounds from the source
        t_full = MDEFeatures(src, include_raw=False).fit(Xs)
        pipe = make_lgb_pipeline(make_lgb()).fit(t_full.transform(Xs), ys)
        Ft = compute_mde(clean_for_mde(raw_t), tgt, bounds=t_full.bounds_)[t_full.mde_columns_]
        p = pipe.predict_proba(Ft.to_numpy(dtype=float))[:, 1]
        rows.append({"source": src, "target": tgt, "kind": "transfer", "n_train": len(ys), "n_test": len(yt),
                     **full_metrics(yt, (p >= 0.5).astype(int), p)})
        print(rows[-2], rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "transfer.csv", index=False)

# ── Noise robustness (entropy-only, LightGBM, 5-fold CV, noise added at inference) ─────────
if wanted("robustness"):
    print("\n=== Robustness ===", flush=True)
    EPS_LEVELS = [0.0, 0.05, 0.10, 0.25]
    rows = []
    for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
        _, X, y = data[ds]
        per_eps = {e: [] for e in EPS_LEVELS}
        for fold_i, (tr, te) in enumerate(CV.split(X, y)):
            t = MDEFeatures(ds, include_raw=False).fit(X.iloc[tr])
            Ftr, Fte = t.transform(X.iloc[tr]), t.transform(X.iloc[te])
            pipe = make_lgb_pipeline(make_lgb()).fit(Ftr, y[tr])
            sigma = np.nanstd(Ftr, axis=0)
            rng = np.random.RandomState(RANDOM_STATE + fold_i)
            for e in EPS_LEVELS:
                noisy = Fte + rng.normal(0.0, 1.0, Fte.shape) * (e * sigma)
                per_eps[e].append(full_metrics(y[te], pipe.predict(noisy), pipe.predict_proba(noisy)[:, 1]))
        for e in EPS_LEVELS:
            row = {"dataset": ds, "eps": e, **aggregate(per_eps[e])}
            rows.append(row); print({k: row[k] for k in ("dataset", "eps", "f1", "dr", "far", "mcc")}, flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "robustness.csv", index=False)

# ── Entropy-only versus the raw statistics it is derived from (LightGBM, 5-fold CV) ────────
if wanted("simplestats"):
    print("\n=== Simple statistics ===", flush=True)
    rows = []
    for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
        _, X, y = data[ds]
        cols = [c for c in SIMPLE_STATS[ds] if c in X.columns]
        f_simple, f_ent, f_strict, n_ent = [], [], [], None
        for tr, te in CV.split(X, y):
            pipe = make_lgb_pipeline(make_lgb()).fit(X.iloc[tr][cols].to_numpy(dtype=float), y[tr])
            Xte_s = X.iloc[te][cols].to_numpy(dtype=float)
            f_simple.append(full_metrics(y[te], pipe.predict(Xte_s), pipe.predict_proba(Xte_s)[:, 1]))
            t = MDEFeatures(ds, include_raw=False).fit(X.iloc[tr])
            n_ent = len(t.mde_columns_)
            Ftr, Fte = np.asarray(t.transform(X.iloc[tr])), np.asarray(t.transform(X.iloc[te]))
            pipe = make_lgb_pipeline(make_lgb()).fit(Ftr, y[tr])
            f_ent.append(full_metrics(y[te], pipe.predict(Fte), pipe.predict_proba(Fte)[:, 1]))
            keep = [i for i, c in enumerate(t.mde_columns_) if c not in NON_ENTROPY_DESCRIPTORS]
            n_strict = len(keep)
            pipe = make_lgb_pipeline(make_lgb()).fit(Ftr[:, keep], y[tr])
            f_strict.append(full_metrics(y[te], pipe.predict(Fte[:, keep]), pipe.predict_proba(Fte[:, keep])[:, 1]))
        rs = {"dataset": ds, "feature_set": "simple_statistics", "n_feat": len(cols), **aggregate(f_simple)}
        re_ = {"dataset": ds, "feature_set": "entropy_only", "n_feat": n_ent, **aggregate(f_ent)}
        rt = {"dataset": ds, "feature_set": "entropy_strict", "n_feat": n_strict, **aggregate(f_strict)}
        rows += [rs, re_, rt]
        print({k: rs[k] for k in ("dataset", "feature_set", "n_feat", "f1")}, {k: re_[k] for k in ("feature_set", "n_feat", "f1")}, flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "simplestats.csv", index=False)

# ── Profiling (whole pipeline, combined features, 80/20 split, single thread) ──────────────────────────────
if wanted("profiling"):
    print("\n=== Profiling ===", flush=True)
    rows = []
    for ds in ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]:
        _, X, y = data[ds]
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        for model_name, make_pipe, make_clf in [("LightGBM", make_lgb_pipeline, make_lgb), ("RandomForest", make_rf_pipeline, make_rf)]:
            clf = make_clf().set_params(n_jobs=1)  # single-threaded by definition of the profiling table
            pipe = make_pipe(clf, mde=MDEFeatures(ds, include_raw=True))
            t0 = time.perf_counter(); pipe.fit(X.iloc[tr], y[tr]); train_s = time.perf_counter() - t0
            Xte = X.iloc[te]
            t0 = time.perf_counter(); pipe.predict(Xte); infer_us = (time.perf_counter() - t0) / len(te) * 1e6
            # split of the same inference: feature stage (entropy features, imputation, clipping) then classifier
            t0 = time.perf_counter(); Z = pipe[:-1].transform(Xte); feat_us = (time.perf_counter() - t0) / len(te) * 1e6
            t0 = time.perf_counter(); pipe[-1].predict(Z); clf_us = (time.perf_counter() - t0) / len(te) * 1e6
            row = {"dataset": ds, "model": model_name, "n_train": len(tr), "train_s": round(train_s, 2), "infer_us": round(infer_us, 2),
                   "feature_us": round(feat_us, 2), "classifier_us": round(clf_us, 2)}
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "profiling.csv", index=False)
print("\nDone.", flush=True)
