"""
Tabular classifier comparison on combined MDE features, 5-fold CV, four datasets.

Classifiers (all trained on the full training fold under the same fold-local pipeline):
  LightGBM, Random Forest, XGBoost, CatBoost, MLP, TabNet, FT-Transformer

Usage:
  python run_baselines.py                                   # everything
  python run_baselines.py --datasets CICIDS-2017 NSL-KDD --models MLP TabNet FT-Transformer
      # a subset; its rows replace the matching rows of the existing tables (--suffix writes
      # separate files instead, merged later with --merge)
  python run_baselines.py --merge                           # fold *_<suffix>.csv part files into the tables

Outputs:
  results/tables/baselines_comparison.csv, baselines_comparison_folds.csv
"""
import sys, warnings, math, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    make_scorer, matthews_corrcoef, average_precision_score,
    recall_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}", flush=True)

from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper, MDEFeatures
from metrics import full_metrics, aggregate, cv_full_metrics

TABLES.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ALL_DATASETS = ["NSL-KDD", "CICIDS-2017", "CICIDS-2018", "UNSW-NB15"]
ALL_MODELS = ["LightGBM", "Random Forest", "XGBoost", "CatBoost", "MLP", "TabNet", "FT-Transformer"]
META = ["binary_label", "multi_label", "label_name"]
SUBSAMPLE_N = None  # every model trains on the full training fold

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
ap.add_argument("--models", nargs="+", default=ALL_MODELS)
ap.add_argument("--suffix", default="", help="write baselines_comparison_<suffix>.csv part files")
ap.add_argument("--merge", action="store_true", help="merge all part files into the main tables and exit")
ARGS = ap.parse_args()
DATASETS = ARGS.datasets
MODELS = set(ARGS.models)


def merge_into(main_path, part):
    """Replace the (dataset, model) rows of the main table with those of `part`, keeping the model order."""
    if main_path.exists():
        base = pd.read_csv(main_path)
        keys = set(zip(part["dataset"], part["model"]))
        base = base[[(d, m) not in keys for d, m in zip(base["dataset"], base["model"])]]
        out = pd.concat([base, part], ignore_index=True)
    else:
        out = part
    out["_d"] = out["dataset"].map({d: i for i, d in enumerate(ALL_DATASETS)})
    out["_m"] = out["model"].map({m: i for i, m in enumerate(ALL_MODELS)})
    sort_cols = ["_d", "_m"] + (["fold"] if "fold" in out.columns else [])
    out = out.sort_values(sort_cols).drop(columns=["_d", "_m"])
    out.to_csv(main_path, index=False)


if ARGS.merge:
    for stem in ["baselines_comparison", "baselines_comparison_folds"]:
        for part_path in sorted(TABLES.glob(f"{stem}_*.csv")):
            if stem == "baselines_comparison" and part_path.name.startswith("baselines_comparison_folds"):
                continue  # the folds table and its part files belong to the other stem
            merge_into(TABLES / f"{stem}.csv", pd.read_csv(part_path))
            print("merged", part_path.name, flush=True)
    sys.exit(0)


# ── Metric helpers ───────────────────────────────────────────────────────────

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


def metrics_from_scores(s):
    return {
        "f1":    round(s["test_f1_weighted"].mean(),        4),
        "f1_std":round(s["test_f1_weighted"].std(),         4),
        "prec":  round(s["test_precision_weighted"].mean(), 4),
        "rec":   round(s["test_recall_weighted"].mean(),    4),
        "dr":    round(s["test_dr"].mean(),                 4),
        "far":   round(s["test_far"].mean(),                4),
        "acc":   round(s["test_accuracy"].mean(),           4),
        "mcc":   round(s["test_mcc"].mean(),                4),
        "auc":   round(s["test_roc_auc"].mean(),            4),
        "prauc": round(s["test_prauc"].mean(),              4),
    }


# ── Sklearn-compatible pipeline factories ───────────────────────────────────

def _preproc_pipe(clf, ds_name=None):
    """Fold-local pipeline: MDE features from the fold's training portion (when ds_name is
    given), then median imputation, percentile clipping, and the classifier."""
    return Pipeline(([("mde", MDEFeatures(ds_name, include_raw=True))] if ds_name else []) + [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper()),
        ("clf", clf),
    ])


def make_lgb_pipe(ds_name=None):
    return _preproc_pipe(ds_name=ds_name, clf=lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    ))

def make_rf_pipe(ds_name=None):
    return _preproc_pipe(ds_name=ds_name, clf=RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    ))

def make_xgb_pipe(ds_name=None):
    use_gpu = DEVICE.type == "cuda"
    return _preproc_pipe(ds_name=ds_name, clf=xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=1, n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0,
        device="cuda" if use_gpu else "cpu",
    ))

def make_catboost_pipe(ds_name=None):
    return _preproc_pipe(ds_name=ds_name, clf=CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0,
        task_type="CPU",
    ))

def make_mlp_pipe(ds_name=None):
    return _preproc_pipe(ds_name=ds_name, clf=MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300,
        random_state=RANDOM_STATE, early_stopping=True,
    ))


# ── FT-Transformer (Gorishniy et al., 2021) ─────────────────────────────────

class _FTTNet(nn.Module):
    """Feature Tokenizer + Transformer for binary classification."""
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.B = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=max(int(d_token * 4 / 3), n_heads),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 2))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.W + self.B          # (B, F, d)
        cls = self.cls.expand(x.size(0), -1, -1)            # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, F+1, d)
        out = self.transformer(tokens)
        return self.head(out[:, 0])


class FTTransformerClassifier:
    """Sklearn-compatible wrapper for FT-Transformer."""
    def __init__(self, d_token=64, n_heads=4, n_layers=2, dropout=0.1,
                 lr=1e-3, max_epochs=30, batch_size=512, random_state=RANDOM_STATE):
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        n, d = X.shape
        classes = np.unique(y)
        w = compute_class_weight("balanced", classes=classes, y=y)
        weights = torch.FloatTensor(w).to(DEVICE)
        self.model_ = _FTTNet(d, self.d_token, self.n_heads, self.n_layers, self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=self.batch_size, shuffle=True,
        )
        self.model_.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(self.model_(xb), yb).backward()
                optimizer.step()
        self.model_.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, X):
        self.model_.to(DEVICE).eval()
        parts = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                logits = self.model_(torch.FloatTensor(X[i:i + 4096]).to(DEVICE))
                parts.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(parts, axis=0)
        self.model_.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ── Manual CV for models that don't fit sklearn cross_validate ───────────────

def _prep(X_tr, X_te):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cli = PercentileClipper().fit(imp.transform(X_tr))
    return cli.transform(imp.transform(X_tr)), cli.transform(imp.transform(X_te))


CKPT = TABLES / "_ckpt_baselines.csv"


def _ckpt_load(ds_name, model_label):
    """Completed folds of (dataset, model) from the checkpoint file, keyed by fold number."""
    if not CKPT.exists():
        return {}
    c = pd.read_csv(CKPT)
    c = c[(c["dataset"] == ds_name) & (c["model"] == model_label)]
    return {int(r["fold"]): {k: v for k, v in r.items() if k not in ("dataset", "model")} for _, r in c.iterrows()}


def _ckpt_save(ds_name, model_label, m):
    row = pd.DataFrame([{"dataset": ds_name, "model": model_label, **m}])
    row.to_csv(CKPT, mode="a", header=not CKPT.exists(), index=False)


def manual_cv(make_clf, X, y, cv=CV, subsample_n=None, model_label="", ds_name=None):
    """Manual 5-fold CV with the complete metric suite; with a raw-statistics DataFrame and
    ds_name, the MDE features are built inside each fold from the training portion. Every finished
    fold is checkpointed immediately, and folds found in the checkpoint file are reused."""
    folds = []
    done = _ckpt_load(ds_name, model_label)
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        if fold_i + 1 in done:
            m = done[fold_i + 1]
            folds.append(m)
            print(f"    fold {fold_i+1}/5  F1={m['f1']:.4f}  DR={m['dr']:.4f}  (from checkpoint)", flush=True)
            continue
        if isinstance(X, pd.DataFrame):
            mde = MDEFeatures(ds_name, include_raw=True).fit(X.iloc[tr_idx])
            X_tr, X_te = mde.transform(X.iloc[tr_idx]), mde.transform(X.iloc[te_idx])
        else:
            X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if subsample_n and len(X_tr) > subsample_n:
            sss = StratifiedShuffleSplit(1, train_size=subsample_n, random_state=RANDOM_STATE + fold_i)
            sub, _ = next(sss.split(X_tr, y_tr))
            X_tr, y_tr = X_tr[sub], y_tr[sub]
        Xtr, Xte = _prep(X_tr, X_te)
        clf = make_clf()
        clf.fit(Xtr, y_tr)
        m = full_metrics(y_te, clf.predict(Xte), clf.predict_proba(Xte)[:, 1])
        m["fold"] = fold_i + 1
        folds.append(m)
        _ckpt_save(ds_name, model_label, m)
        print(f"    fold {fold_i+1}/5  F1={m['f1']:.4f}  DR={m['dr']:.4f}", flush=True)
    fdf = pd.DataFrame(folds); fdf.insert(0, "model", model_label); fdf.insert(0, "dataset", ds_name); fold_rows.append(fdf)
    return aggregate([{k: v for k, v in f.items() if k != "fold"} for f in folds])


def make_tabnet_factory(n_features):
    def _make():
        return TabNetClassifier(
            n_d=32, n_a=32, n_steps=5, gamma=1.3,
            n_independent=2, n_shared=2,
            momentum=0.02, clip_value=2.0,
            seed=RANDOM_STATE, verbose=0,
        )
    return _make


class TabNetWrapper:
    """Wraps pytorch-tabnet to match the manual_cv predict_proba interface."""
    def __init__(self, n_features):
        self._n = n_features
        self._clf = None

    def fit(self, X, y):
        self._clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5, gamma=1.3,
            n_independent=2, n_shared=2,
            momentum=0.02, clip_value=2.0,
            seed=RANDOM_STATE, verbose=0,
            device_name="cuda" if DEVICE.type == "cuda" else "cpu",
        )
        # Use a small validation split inside TabNet for early stopping
        sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=RANDOM_STATE)
        tr, vl = next(sss.split(X, y))
        self._clf.fit(
            X[tr].astype(np.float32), y[tr],
            eval_set=[(X[vl].astype(np.float32), y[vl])],
            eval_metric=["accuracy"], max_epochs=50, patience=10,
            batch_size=1024, virtual_batch_size=128,
        )
        return self

    def predict(self, X):
        return self._clf.predict(X.astype(np.float32))

    def predict_proba(self, X):
        return self._clf.predict_proba(X.astype(np.float32))


# ── Classifier definitions: (name, make_fn, use_manual_cv, subsample_n) ─────

CLASSIFIERS = [
    ("LightGBM",        make_lgb_pipe,      False, None),
    ("Random Forest",   make_rf_pipe,       False, None),
    ("XGBoost",         make_xgb_pipe,      False, None),
    ("CatBoost",        make_catboost_pipe, False, None),
    ("MLP",             make_mlp_pipe,      True,  SUBSAMPLE_N),
]


# ── Main loop ────────────────────────────────────────────────────────────────

rows = []
fold_rows = []
for ds_name in DATASETS:
    print(f"\n{'='*65}", flush=True)
    print(f"  DATASET: {ds_name}", flush=True)
    print(f"{'='*65}", flush=True)
    raw   = load_dataset(ds_name)
    df    = clean(raw)
    X     = df.drop(columns=META, errors="ignore").select_dtypes(include=[np.number])
    y     = df["binary_label"].values
    n_feat = len(MDEFeatures(ds_name, include_raw=True).fit(X).feature_names_out_)
    print(f"  combined: {n_feat} features, {len(y):,} samples", flush=True)

    # ── Sklearn-pipeline classifiers ─────────────────────────────────────────
    for clf_name, make_fn, use_manual, sub_n in CLASSIFIERS:
        if clf_name not in MODELS:
            continue
        print(f"\n  [{clf_name}]", flush=True)
        if use_manual:
            m = manual_cv(make_fn, X, y, subsample_n=sub_n, model_label=clf_name, ds_name=ds_name)
        else:
            def _fp(tr, te, make_fn=make_fn):
                pipe = make_fn(ds_name)
                pipe.fit(X.iloc[tr], y[tr])
                return pipe.predict(X.iloc[te]), pipe.predict_proba(X.iloc[te])[:, 1]
            m, folds = cv_full_metrics(_fp, X, y, CV, label=clf_name)
            folds.insert(0, "model", clf_name); folds.insert(0, "dataset", ds_name); fold_rows.append(folds)
        row = {"dataset": ds_name, "model": clf_name, **m}
        print(f"  → F1={m['f1']:.4f} (±{m['f1_std']:.4f})  "
              f"DR={m['dr']:.4f}  FAR={m['far']:.4f}  MCC={m['mcc']:.4f}", flush=True)
        rows.append(row)

    # ── TabNet ───────────────────────────────────────────────────────────────
    if "TabNet" in MODELS:
      print(f"\n  [TabNet]", flush=True)
      tabnet_m = manual_cv(
        lambda nf=n_feat: TabNetWrapper(nf), X, y,
        subsample_n=SUBSAMPLE_N, model_label="TabNet", ds_name=ds_name,
    )
      row_tn = {"dataset": ds_name, "model": "TabNet", **tabnet_m}
      print(f"  → F1={tabnet_m['f1']:.4f} (±{tabnet_m['f1_std']:.4f})  "
            f"DR={tabnet_m['dr']:.4f}  FAR={tabnet_m['far']:.4f}  "
            f"MCC={tabnet_m['mcc']:.4f}", flush=True)
      rows.append(row_tn)

    # ── FT-Transformer ───────────────────────────────────────────────────────
    if "FT-Transformer" in MODELS:
      print(f"\n  [FT-Transformer]", flush=True)
      ftt_m = manual_cv(
        lambda nf=n_feat: FTTransformerClassifier(random_state=RANDOM_STATE),
        X, y, subsample_n=SUBSAMPLE_N, model_label="FT-Transformer", ds_name=ds_name,
    )
      row_ftt = {"dataset": ds_name, "model": "FT-Transformer", **ftt_m}
      print(f"  → F1={ftt_m['f1']:.4f} (±{ftt_m['f1_std']:.4f})  "
            f"DR={ftt_m['dr']:.4f}  FAR={ftt_m['far']:.4f}  "
            f"MCC={ftt_m['mcc']:.4f}", flush=True)
      rows.append(row_ftt)

df_out = pd.DataFrame(rows)
df_folds = pd.concat(fold_rows, ignore_index=True)
if ARGS.suffix:
    df_out.to_csv(TABLES / f"baselines_comparison_{ARGS.suffix}.csv", index=False)
    df_folds.to_csv(TABLES / f"baselines_comparison_folds_{ARGS.suffix}.csv", index=False)
    print(f"\nSaved part files with suffix {ARGS.suffix}; run --merge to fold them in", flush=True)
else:
    merge_into(TABLES / "baselines_comparison.csv", df_out)
    merge_into(TABLES / "baselines_comparison_folds.csv", df_folds)
    print(f"\nSaved: {TABLES / 'baselines_comparison.csv'}", flush=True)
