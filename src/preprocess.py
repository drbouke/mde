import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from config import DATASETS, RANDOM_STATE


def _load_cicids17(cfg):
    folder = Path(cfg["path"])
    frames = []
    for f in folder.glob("*.csv"):
        try:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
            # Fix duplicate column name
            cols = pd.Series(df.columns.str.strip())
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            df.columns = new_cols
            frames.append(df)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return pd.concat(frames, ignore_index=True)


def load_dataset(name):
    cfg = DATASETS[name]
    print(f"\n[load] {name}")

    if cfg["type"] == "cicids17":
        df = _load_cicids17(cfg)
    else:
        nrows = cfg.get("nrows", None)
        df = pd.read_csv(cfg["path"], low_memory=False, nrows=nrows)

    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    label_col = cfg["label_col"]

    # Remove header lines repeated inside the file (rows whose value equals the column name)
    # and coerce numeric columns that such rows turned into text.
    probe = next((c for c in df.columns if c != label_col), None)
    if probe is not None and df[probe].dtype == object:
        hdr = df[probe].astype(str).str.strip() == probe
        if hdr.any():
            df = df.loc[~hdr].reset_index(drop=True)
            print(f"  Dropped {int(hdr.sum())} repeated header rows")
    for c in df.columns:
        if c == label_col or df[c].dtype != object:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() >= 0.99 * df[c].notna().sum():
            df[c] = conv

    if "sample_n" in cfg and len(df) > cfg["sample_n"]:
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(cfg["sample_n"] * len(x) / len(df)))),
                random_state=RANDOM_STATE,
            )
        ).reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows (stratified)")

    # Drop identifier / leakage columns
    drop = [c for c in cfg.get("drop_cols", []) if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.strip()

    # Binary label: 0=benign, 1=attack
    benign = cfg["benign_labels"]
    df["binary_label"] = (~df[label_col].isin([str(b) for b in benign]) &
                          ~df[label_col].isin(benign)).astype(int)

    # Multi-class label (keep original, encoded)
    le = LabelEncoder()
    df["multi_label"] = le.fit_transform(df[label_col].astype(str))
    df["label_name"] = df[label_col].astype(str)

    # Drop the original label column so it cannot leak into feature matrices.
    # All label information is preserved in binary_label / multi_label / label_name.
    df.drop(columns=[label_col], inplace=True, errors="ignore")

    print(f"  Rows: {len(df):,}  |  Benign: {(df['binary_label']==0).sum():,}  "
          f"|  Attack: {(df['binary_label']==1).sum():,}")
    print(f"  Attack types: {df['label_name'].nunique()}")
    return df


def clean(df, label_cols=("binary_label", "multi_label", "label_name")):
    """
    Encode categoricals, replace inf with NaN, and re-attach label metadata.
    Does NOT impute NaN or clip outliers — those steps are performed
    fold-locally inside sklearn Pipelines to prevent cross-fold leakage.
    Use clean_for_mde() when NaN-free columns are required for entropy formulas.
    """
    meta = df[list(label_cols)].copy()
    df = df.drop(columns=list(label_cols), errors="ignore")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    # Replace inf/-inf with NaN — actual imputation happens fold-locally
    df = df.replace([np.inf, -np.inf], np.nan)

    df = pd.concat([df, meta], axis=1)
    return df


def clean_for_mde(df, label_cols=("binary_label", "multi_label", "label_name"),
                  stats=None, return_stats=False):
    """
    Returns a NaN-free copy of df for MDE entropy formula input. Missing and infinite
    values are replaced by column medians taken from `stats` when given (the training
    portion of a split), otherwise from df itself. With return_stats=True the medians
    used are returned as well, so a test portion can be imputed with training medians.
    The returned df is meant only for computing MDE features, not as a model input.
    """
    meta = df[list(label_cols)].copy()
    df_num = df.drop(columns=list(label_cols), errors="ignore")

    cat_cols = df_num.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df_num[c] = LabelEncoder().fit_transform(df_num[c].astype(str))

    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    if stats is None:
        stats = df_num.median(numeric_only=True)
    df_num = df_num.fillna(stats)

    df_out = pd.concat([df_num, meta], axis=1)
    return (df_out, stats) if return_stats else df_out
