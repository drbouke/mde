"""
Per-class detection rates: CICIDS-2017 time-split (debiased), CICIDS-2018 hold-out, NSL-KDD hold-out.

Outputs:
  results/tables/perclass_timesplit.csv
  results/tables/perclass_cicids18.csv
  results/tables/perclass_nslkdd.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from config import DATA, TABLES, RANDOM_STATE
from preprocess import load_dataset, clean, clean_for_mde
from entropy_features import compute_mde, build_feature_sets
from fold_pipeline import PercentileClipper

TABLES.mkdir(parents=True, exist_ok=True)
AUROC_THRESHOLD = 0.99

DROP_COLS = ["Flow ID", "Source IP", "Source Port",
             "Destination IP", "Destination Port", "Timestamp"]


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE, verbose=-1,
    )


def load_cicids_files(file_list, sample_n=None, max_per_file=100_000):
    """Load CICIDS CSV files; process and drop columns per-file to limit memory."""
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
            df["multi_label"]  = LabelEncoder().fit_transform(df[lc].astype(str))
            df["label_name"]   = df[lc]
            df.drop(columns=[lc], inplace=True, errors="ignore")
            for c in DROP_COLS:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            if max_per_file and len(df) > max_per_file:
                df = df.sample(n=max_per_file, random_state=RANDOM_STATE)
            frames.append(df)
        except Exception as e:
            print(f"  Skip {fp.name}: {e}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    if sample_n and len(df) > sample_n:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_n,
                                     random_state=RANDOM_STATE)
        idx, _ = next(sss.split(df, df["binary_label"]))
        df = df.iloc[idx].reset_index(drop=True)
    return df


def align(X_tr, feat_tr, X_te, feat_te):
    feat_tr, feat_te = list(feat_tr), list(feat_te)
    common = [f for f in feat_tr if f in feat_te]
    return (X_tr[:, [feat_tr.index(f) for f in common]],
            X_te[:, [feat_te.index(f) for f in common]],
            common)


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
        print(f"    removed: {name:45s} AUC={auc}", flush=True)
    print(f"    Kept {len(keep)}/{len(feat_names)} features after debiasing", flush=True)
    return [i for i, *_ in keep]


def preprocess_fit_transform(X_tr, X_te):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cli = PercentileClipper().fit(imp.transform(X_tr))
    return cli.transform(imp.transform(X_tr)), cli.transform(imp.transform(X_te))


# ── Part A: CICIDS-2017 time-split per-category (debiased, same as timesplit table) ──

print("\n=== CICIDS-2017 Time-Split Per-Category (debiased) ===", flush=True)
cic17_dir   = DATA / "CICIDS2017" / "TrafficLabelling_"
fri_files   = sorted(f for f in cic17_dir.glob("*.csv") if "Friday"   in f.name)
train_files = sorted(f for f in cic17_dir.glob("*.csv") if "Friday" not in f.name)

print("  Loading Mon-Thu (train, 150K)...", flush=True)
df_tr_raw = load_cicids_files(train_files, sample_n=150_000)
print("  Loading Friday (test, 100K)...", flush=True)
df_te_raw = load_cicids_files(fri_files,   sample_n=100_000)

df_tr = clean(df_tr_raw)
df_te = clean(df_te_raw)
mde_input_tr = clean_for_mde(df_tr_raw)
mde_input_te = clean_for_mde(df_te_raw)
mde_tr = compute_mde(mde_input_tr, "CICIDS-2017")
mde_te = compute_mde(mde_input_te, "CICIDS-2017", fit_df=mde_input_tr)
fsets_tr = build_feature_sets(df_tr, mde_tr)
fsets_te = build_feature_sets(df_te, mde_te)
y_tr = fsets_tr["conventional"][1]
y_te = fsets_te["conventional"][1]

print("\n  Debiasing conventional features...", flush=True)
X_tr_conv, X_te_conv, feat_conv = align(
    fsets_tr["conventional"][0], fsets_tr["conventional"][2],
    fsets_te["conventional"][0], fsets_te["conventional"][2])
keep_conv = debias_mask(X_tr_conv, y_tr, feat_conv)
X_tr_conv, X_te_conv = X_tr_conv[:, keep_conv], X_te_conv[:, keep_conv]

print("\n  Debiasing entropy-only features...", flush=True)
X_tr_ent, X_te_ent, feat_ent = align(
    fsets_tr["entropy_only"][0], fsets_tr["entropy_only"][2],
    fsets_te["entropy_only"][0], fsets_te["entropy_only"][2])
keep_ent = debias_mask(X_tr_ent, y_tr, feat_ent)
X_tr_ent, X_te_ent = X_tr_ent[:, keep_ent], X_te_ent[:, keep_ent]

X_tr_comb = np.concatenate([X_tr_conv, X_tr_ent], axis=1)
X_te_comb = np.concatenate([X_te_conv, X_te_ent], axis=1)

Xtr_comb, Xte_comb = preprocess_fit_transform(X_tr_comb, X_te_comb)

clf17 = make_lgb()
clf17.fit(Xtr_comb, y_tr)
y_pred17 = clf17.predict(Xte_comb)

label_names_te = df_te_raw["label_name"].values
rows_ts = []
for cat in sorted(df_te_raw["label_name"].unique()):
    mask = label_names_te == cat
    n = mask.sum()
    if n < 5:
        continue
    if cat == "BENIGN":
        far = (y_pred17[mask] == 1).mean()
        rows_ts.append({"category": cat, "n_test": int(n), "DR": None, "FAR": round(far, 4)})
    else:
        dr = (y_pred17[mask] == 1).mean()
        rows_ts.append({"category": cat, "n_test": int(n), "DR": round(dr, 4), "FAR": None})

df_ts = pd.DataFrame(rows_ts)
print(df_ts.to_string(index=False), flush=True)
df_ts.to_csv(TABLES / "perclass_timesplit.csv", index=False)

# ── Part B: CICIDS-2018 per-category hold-out ─────────────────────────────────

print("\n=== CICIDS-2018 Per-Category (20% hold-out) ===", flush=True)
raw18 = load_dataset("CICIDS-2018")
df18  = clean(raw18)
mde18 = compute_mde(clean_for_mde(raw18), "CICIDS-2018")
fsets18 = build_feature_sets(df18, mde18)
X18, y18, _ = fsets18["combined"]

X_tr18, X_te18, y_tr18, y_te18, idx_tr18, idx_te18 = train_test_split(
    X18, y18, np.arange(len(y18)), test_size=0.2, stratify=y18,
    random_state=RANDOM_STATE)

Xtr18, Xte18 = preprocess_fit_transform(X_tr18, X_te18)
clf18 = make_lgb()
clf18.fit(Xtr18, y_tr18)
y_pred18 = clf18.predict(Xte18)

label_names18 = df18["label_name"].values[idx_te18]
rows18 = []
for cat in sorted(set(label_names18)):
    mask = label_names18 == cat
    n = mask.sum()
    if n < 5:
        continue
    if cat == "0":
        far = (y_pred18[mask] == 1).mean()
        rows18.append({"category": "Benign", "n_test": int(n), "DR": None, "FAR": round(far, 4)})
    else:
        dr = (y_pred18[mask] == 1).mean()
        rows18.append({"category": cat, "n_test": int(n), "DR": round(dr, 4), "FAR": None})

df18_out = pd.DataFrame(rows18)
print(df18_out.to_string(index=False), flush=True)
df18_out.to_csv(TABLES / "perclass_cicids18.csv", index=False)

# ── Part C: NSL-KDD per-category hold-out ────────────────────────────────────

print("\n=== NSL-KDD Per-Category (20% hold-out) ===", flush=True)
rawnsl = load_dataset("NSL-KDD")
dfnsl  = clean(rawnsl)
mdensl = compute_mde(clean_for_mde(rawnsl), "NSL-KDD")
fsetsnsl = build_feature_sets(dfnsl, mdensl)
Xnsl, ynsl, _ = fsetsnsl["combined"]

X_trnsl, X_tensl, y_trnsl, y_tensl, idx_trnsl, idx_tensl = train_test_split(
    Xnsl, ynsl, np.arange(len(ynsl)), test_size=0.2, stratify=ynsl,
    random_state=RANDOM_STATE)

Xtrnsl, Xtensl = preprocess_fit_transform(X_trnsl, X_tensl)
clfnsl = make_lgb()
clfnsl.fit(Xtrnsl, y_trnsl)
y_prednsl = clfnsl.predict(Xtensl)

label_namesnsl = dfnsl["label_name"].values[idx_tensl]
rows_nsl = []
for cat in sorted(set(label_namesnsl)):
    mask = label_namesnsl == cat
    n = mask.sum()
    if n < 5:
        continue
    is_benign = (dfnsl["binary_label"].values[idx_tensl][mask] == 0).all()
    if is_benign:
        far = (y_prednsl[mask] == 1).mean()
        rows_nsl.append({"category": cat, "n_test": int(n), "DR": None, "FAR": round(far, 4)})
    else:
        dr = (y_prednsl[mask] == 1).mean()
        rows_nsl.append({"category": cat, "n_test": int(n), "DR": round(dr, 4), "FAR": None})

df_nsl = pd.DataFrame(rows_nsl)
print(df_nsl.to_string(index=False), flush=True)
df_nsl.to_csv(TABLES / "perclass_nslkdd.csv", index=False)

print("\n=== Done ===", flush=True)
