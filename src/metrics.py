"""
Complete binary-classification metric suite shared by every experiment script.

full_metrics(y_true, y_pred, y_prob) returns one flat dict with the confusion-matrix counts
(tn, fp, fn, tp), accuracy, weighted and macro precision/recall/F1, attack-class
precision/recall/F1, detection rate (DR), false alarm rate (FAR), false negative rate, true
negative rate, balanced accuracy, MCC, ROC-AUC, and PR-AUC. aggregate(fold_dicts) averages the
rate metrics over folds (mean and std) and sums the counts. cv_full_metrics runs a caller-supplied
fit-and-predict function on every fold of a splitter and returns the aggregate together with a
per-fold table.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

COUNT_KEYS = ("n", "n_attack", "n_benign", "tn", "fp", "fn", "tp")


def _div(a, b):
    return float(a) / b if b > 0 else 0.0


def full_metrics(y_true, y_pred, y_prob=None):
    """All metrics for one evaluation set; rates rounded to 4 decimals, counts as integers."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
    m = {
        "n": int(len(y_true)), "n_attack": int(tp + fn), "n_benign": int(tn + fp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "prec_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "prec_attack": _div(tp, tp + fp),
        "rec_attack": _div(tp, tp + fn),
        "f1_attack": _div(2 * tp, 2 * tp + fp + fn),
        "dr": _div(tp, tp + fn),
        "far": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "tnr": _div(tn, tn + fp),
        "bal_acc": 0.5 * (_div(tp, tp + fn) + _div(tn, tn + fp)),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["auc"] = roc_auc_score(y_true, y_prob)
        m["prauc"] = average_precision_score(y_true, y_prob, pos_label=1)
    else:
        m["auc"] = float("nan")
        m["prauc"] = float("nan")
    return {k: (v if k in COUNT_KEYS else round(float(v), 4)) for k, v in m.items()}


def aggregate(fold_metrics):
    """Mean and std of every rate metric over folds (keys <m> and <m>_std), counts summed."""
    out = {}
    keys = [k for k in fold_metrics[0] if k not in COUNT_KEYS]
    for k in keys:
        vals = np.array([f[k] for f in fold_metrics], dtype=float)
        out[k] = round(float(np.nanmean(vals)), 4)
        out[f"{k}_std"] = round(float(np.nanstd(vals)), 4)
    for k in COUNT_KEYS:
        out[k] = int(sum(f[k] for f in fold_metrics))
    out["n_folds"] = len(fold_metrics)
    return out


def cv_full_metrics(fit_predict, X, y, cv, label="", verbose=True):
    """
    fit_predict(tr_idx, te_idx) -> (y_pred, y_prob) for one fold, fitted on tr_idx only.
    Returns (aggregate dict, per-fold DataFrame).
    """
    folds = []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        y_pred, y_prob = fit_predict(tr, te)
        m = full_metrics(y[te], y_pred, y_prob)
        m["fold"] = i + 1
        folds.append(m)
        if verbose:
            print(f"    {label} fold {i + 1}/{cv.get_n_splits()}  F1={m['f1']:.4f}  DR={m['dr']:.4f}  "
                  f"FAR={m['far']:.4f}  MCC={m['mcc']:.4f}", flush=True)
    summary = aggregate([{k: v for k, v in f.items() if k != "fold"} for f in folds])
    return summary, pd.DataFrame(folds)
