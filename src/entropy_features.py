"""
Multi-level Distributional Entropy (MDE) feature engineering.

Novel contribution: computes entropy analytically from pre-aggregated flow
statistics, eliminating the need for raw packet sequences. Three entropy
levels are computed per flow:
  L1 - Analytical Differential Entropy (ADE): intra-flow Gaussian entropy
       from mean/std of packet lengths and inter-arrival times.
  L2 - Cross-directional Jensen-Shannon Divergence (JSD): captures the
       statistical asymmetry between forward and backward traffic, a signal
       known to differ between benign bidirectional sessions and attacks
       (e.g., DDoS, scanning) dominated by unidirectional bursts.
  L3 - Flag-pattern Shannon Entropy: uncertainty across TCP control flags,
       distinguishing stealthy flag-manipulation attacks from normal traffic.
"""

import numpy as np
import pandas as pd

EPS = 1e-9


# ── Core entropy primitives ──────────────────────────────────────────────────

def ade_gaussian(std):
    """Differential entropy of N(mu, sigma^2): 0.5*ln(2*pi*e*sigma^2)."""
    s = np.maximum(np.abs(std), EPS)
    return 0.5 * np.log(2 * np.pi * np.e * s ** 2)


def ade_uniform(lo, hi):
    """Differential entropy of Uniform[lo, hi]: ln(hi - lo)."""
    width = np.maximum(hi - lo, EPS)
    return np.log(width)


def binary_shannon(p):
    """Binary Shannon entropy H(p, 1-p)."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def kl_gaussian(mu1, s1, mu2, s2):
    """KL divergence KL(N1 || N2) for univariate Gaussians."""
    s1 = np.maximum(np.abs(s1), EPS)
    s2 = np.maximum(np.abs(s2), EPS)
    return np.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5


def jsd_gaussian(mu1, s1, mu2, s2):
    """
    Jensen-Shannon divergence between two univariate Gaussians.
    Approximated via the moment-matched mixture mean:
      M = 0.5*N1 + 0.5*N2  =>  mu_M = mean(mu1,mu2),
      sigma_M^2 = 0.5*(s1^2 + s2^2) + 0.25*(mu1-mu2)^2
    JSD = 0.5*KL(N1||M) + 0.5*KL(N2||M)  (bounded in [0, ln2])
    """
    mu_m = 0.5 * (mu1 + mu2)
    s_m = np.sqrt(np.maximum(0.5 * (s1 ** 2 + s2 ** 2) + 0.25 * (mu1 - mu2) ** 2, EPS))
    jsd = 0.5 * kl_gaussian(mu1, s1, mu_m, s_m) + 0.5 * kl_gaussian(mu2, s2, mu_m, s_m)
    return np.clip(jsd, 0, np.log(2))


def flag_entropy(flag_df):
    """
    Shannon entropy over TCP flag counts per flow.
    flag_df : DataFrame of non-negative flag count columns.
    Returns Series of per-row entropy values.
    """
    total = flag_df.sum(axis=1).replace(0, np.nan)
    probs = flag_df.div(total, axis=0).fillna(0).clip(lower=EPS)
    return -(probs * np.log(probs)).sum(axis=1)


# ── Dataset-aware MDE computation ────────────────────────────────────────────

def _mde_cicids(df, fit_df=None):
    """MDE for CICIDS-2017 / CICIDS-2018 flow feature schema."""
    F = pd.DataFrame(index=df.index)

    # Column name variants (2017 vs 2018 differ slightly)
    def col(*candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series(np.zeros(len(df)), index=df.index)

    fwd_pkt_mean = col("Fwd Packet Length Mean", "Fwd Pkt Len Mean")
    fwd_pkt_std  = col("Fwd Packet Length Std",  "Fwd Pkt Len Std")
    fwd_pkt_max  = col("Fwd Packet Length Max",  "Fwd Pkt Len Max")
    fwd_pkt_min  = col("Fwd Packet Length Min",  "Fwd Pkt Len Min")
    bwd_pkt_mean = col("Bwd Packet Length Mean", "Bwd Pkt Len Mean")
    bwd_pkt_std  = col("Bwd Packet Length Std",  "Bwd Pkt Len Std")
    bwd_pkt_max  = col("Bwd Packet Length Max",  "Bwd Pkt Len Max")
    bwd_pkt_min  = col("Bwd Packet Length Min",  "Bwd Pkt Len Min")

    fwd_iat_mean = col("Fwd IAT Mean")
    fwd_iat_std  = col("Fwd IAT Std")
    bwd_iat_mean = col("Bwd IAT Mean")
    bwd_iat_std  = col("Bwd IAT Std")

    flow_bytes = col("Flow Bytes/s", "Flow Byts/s")
    tot_fwd    = col("Total Fwd Packets", "Tot Fwd Pkts").replace(0, EPS)
    tot_bwd    = col("Total Backward Packets", "Tot Bwd Pkts").replace(0, EPS)

    # L1 – Analytical Differential Entropy
    F["ade_fwd_pkt_gauss"]  = ade_gaussian(fwd_pkt_std)
    F["ade_bwd_pkt_gauss"]  = ade_gaussian(bwd_pkt_std)
    F["ade_fwd_iat_gauss"]  = ade_gaussian(fwd_iat_std)
    F["ade_bwd_iat_gauss"]  = ade_gaussian(bwd_iat_std)
    F["ade_fwd_pkt_range"]  = ade_uniform(fwd_pkt_min, fwd_pkt_max)
    F["ade_bwd_pkt_range"]  = ade_uniform(bwd_pkt_min, bwd_pkt_max)

    # L2 – Cross-directional JSD
    F["jsd_pkt_len"]  = jsd_gaussian(fwd_pkt_mean, fwd_pkt_std,
                                     bwd_pkt_mean, bwd_pkt_std)
    F["jsd_iat"]      = jsd_gaussian(fwd_iat_mean, fwd_iat_std,
                                     bwd_iat_mean, bwd_iat_std)

    # Directional traffic fraction → binary Shannon entropy
    pkt_frac = tot_fwd / (tot_fwd + tot_bwd)
    F["dir_entropy_pkts"] = binary_shannon(pkt_frac)

    # Log-scale behavioral feature (not an information-theoretic entropy quantity)
    F["log_byte_rate"] = np.log1p(np.abs(flow_bytes))

    # L3 – Flag Shannon entropy
    flag_cols = [c for c in ["FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                              "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                              "CWE Flag Count", "ECE Flag Count",
                              "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
                              "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
                              "CWE Flag Count", "ECE Flag Cnt"]
                 if c in df.columns]
    seen_flags = []
    flag_frame_cols = []
    for c in flag_cols:
        base = c.replace(" Cnt", "").replace(" Count", "")
        if base not in seen_flags:
            seen_flags.append(base)
            flag_frame_cols.append(c)
    if flag_frame_cols:
        F["flag_entropy"] = flag_entropy(df[flag_frame_cols].clip(lower=0))
    else:
        F["flag_entropy"] = 0.0

    # Composite MDE score: min-max normalized using training-set statistics.
    # fit_df provides training-corpus bounds; None falls back to self (e.g. full-dataset CV).
    ent_cols = ["ade_fwd_pkt_gauss", "ade_bwd_pkt_gauss", "ade_fwd_iat_gauss",
                "ade_bwd_iat_gauss", "jsd_pkt_len", "jsd_iat", "flag_entropy"]
    _fit = _mde_cicids(fit_df) if fit_df is not None else F
    lo, hi = _fit[ent_cols].min(), _fit[ent_cols].max()
    F["mde_score"] = ((F[ent_cols] - lo) / (hi - lo + EPS)).mean(axis=1)

    return F


def _mde_unsw(df, fit_df=None):
    """MDE for UNSW-NB15 schema."""
    F = pd.DataFrame(index=df.index)

    sbytes = df.get("sbytes", pd.Series(EPS, index=df.index)).replace(0, EPS)
    dbytes = df.get("dbytes", pd.Series(EPS, index=df.index)).replace(0, EPS)
    spkts  = df.get("Spkts",  pd.Series(EPS, index=df.index)).replace(0, EPS)
    dpkts  = df.get("Dpkts",  pd.Series(EPS, index=df.index)).replace(0, EPS)
    smeansz = df.get("smeansz", pd.Series(0, index=df.index))
    dmeansz = df.get("dmeansz", pd.Series(0, index=df.index))
    sjit   = df.get("Sjit",   pd.Series(EPS, index=df.index))
    djit   = df.get("Djit",   pd.Series(EPS, index=df.index))
    sintpkt = df.get("Sintpkt", pd.Series(EPS, index=df.index))
    dintpkt = df.get("Dintpkt", pd.Series(EPS, index=df.index))
    sload  = df.get("Sload",  pd.Series(EPS, index=df.index)).replace(0, EPS)
    dload  = df.get("Dload",  pd.Series(EPS, index=df.index)).replace(0, EPS)

    # L1 – ADE: use jitter as std proxy for IAT, mean packet size
    F["ade_src_iat"]     = ade_gaussian(sjit)
    F["ade_dst_iat"]     = ade_gaussian(djit)
    F["ade_pkt_range"]   = ade_uniform(
        df.get("smeansz", pd.Series(0, index=df.index)) * 0,  # approx min=0
        smeansz + EPS
    )

    # L2 – JSD src vs dst
    F["jsd_pkt_sz"]  = jsd_gaussian(smeansz, sjit, dmeansz, djit)
    F["jsd_iat"]     = jsd_gaussian(sintpkt, sjit, dintpkt, djit)

    # Directional entropy
    byte_frac = sbytes / (sbytes + dbytes)
    pkt_frac  = spkts  / (spkts  + dpkts)
    F["dir_entropy_bytes"] = binary_shannon(byte_frac)
    F["dir_entropy_pkts"]  = binary_shannon(pkt_frac)

    # Log-scale behavioral features (not information-theoretic entropy quantities)
    F["log_sload"] = np.log1p(np.abs(sload))
    F["log_dload"] = np.log1p(np.abs(dload))

    # L3 – no flag columns available in UNSW → use TTL asymmetry as proxy
    sttl = df.get("sttl", pd.Series(0, index=df.index))
    dttl = df.get("dttl", pd.Series(0, index=df.index))
    ttl_range = np.maximum(np.abs(sttl - dttl), EPS)
    F["ttl_asym_entropy"] = np.log(ttl_range)

    ent_cols = ["ade_src_iat", "ade_dst_iat", "jsd_pkt_sz",
                "jsd_iat", "dir_entropy_bytes", "dir_entropy_pkts"]
    _fit = _mde_unsw(fit_df) if fit_df is not None else F
    lo, hi = _fit[ent_cols].min(), _fit[ent_cols].max()
    F["mde_score"] = ((F[ent_cols] - lo) / (hi - lo + EPS)).mean(axis=1)

    return F


def _mde_kdd(df, fit_df=None):
    """MDE for NSL-KDD schema."""
    F = pd.DataFrame(index=df.index)

    src_bytes = df.get("src_bytes", pd.Series(EPS, index=df.index)).replace(0, EPS)
    dst_bytes = df.get("dst_bytes", pd.Series(EPS, index=df.index)).replace(0, EPS)
    count     = df.get("count",     pd.Series(1, index=df.index)).replace(0, 1)
    srv_count = df.get("srv_count", pd.Series(1, index=df.index)).replace(0, 1)

    # Rate-based features in NSL-KDD are already probability-like [0,1]
    rate_feats = ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                  "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"]
    avail = [c for c in rate_feats if c in df.columns]

    # L1 – treat connection error/rate vector as distribution; compute entropy
    if avail:
        rate_df = df[avail].clip(lower=EPS).fillna(EPS)
        row_sum = rate_df.sum(axis=1).replace(0, EPS)
        probs   = rate_df.div(row_sum, axis=0)
        F["conn_state_entropy"] = -(probs * np.log(probs.clip(lower=EPS))).sum(axis=1)
    else:
        F["conn_state_entropy"] = 0.0

    # L2 – byte directionality entropy
    byte_frac = src_bytes / (src_bytes + dst_bytes)
    F["dir_entropy_bytes"] = binary_shannon(byte_frac)

    # Service diversity entropy
    srv_frac = srv_count / count
    srv_frac = srv_frac.clip(EPS, 1 - EPS)
    F["srv_diversity_entropy"] = binary_shannon(srv_frac)

    # Log-scale behavioral features (not information-theoretic entropy quantities)
    F["log_src_bytes"] = np.log1p(src_bytes)
    F["log_dst_bytes"] = np.log1p(dst_bytes)
    F["byte_asym_jsd"] = jsd_gaussian(
        np.log1p(src_bytes), np.log1p(src_bytes) * 0.1,
        np.log1p(dst_bytes), np.log1p(dst_bytes) * 0.1
    )

    ent_cols = ["conn_state_entropy", "dir_entropy_bytes", "srv_diversity_entropy"]
    _fit = _mde_kdd(fit_df) if fit_df is not None else F
    lo, hi = _fit[ent_cols].min(), _fit[ent_cols].max()
    F["mde_score"] = ((F[ent_cols] - lo) / (hi - lo + EPS)).mean(axis=1)

    return F


# ── Dispatcher ───────────────────────────────────────────────────────────────

DATASET_MDE = {
    "NSL-KDD":     _mde_kdd,
    "CICIDS-2017": _mde_cicids,
    "CICIDS-2018": _mde_cicids,
    "UNSW-NB15":   _mde_unsw,
}


def compute_mde(df, dataset_name, fit_df=None):
    """Compute MDE features for df.

    fit_df: training-set DataFrame used to derive mde_score normalization bounds.
            Pass the training corpus for temporal/hold-out experiments so test-set
            entropy values are normalized against training statistics. When None,
            normalization uses df itself (appropriate for CV with tree-based models,
            which are invariant to monotonic feature rescaling).
    """
    fn = DATASET_MDE[dataset_name]
    mde = fn(df, fit_df=fit_df)
    mde = mde.replace([np.inf, -np.inf], np.nan).fillna(0)
    return mde


def build_feature_sets(df, mde, meta_cols=("binary_label", "multi_label", "label_name")):
    """
    Returns three feature matrices for the ablation study:
      conventional  – original numerical features only
      entropy_only  – MDE features only
      combined      – both together
    """
    drop = list(meta_cols)
    conv_df = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    ent_df  = mde.select_dtypes(include=[np.number])

    y = df["binary_label"].values

    return {
        "conventional":  (conv_df.values, y, list(conv_df.columns)),
        "entropy_only":  (ent_df.values,  y, list(ent_df.columns)),
        "combined":      (pd.concat([conv_df, ent_df], axis=1).values,
                         y, list(conv_df.columns) + list(ent_df.columns)),
    }
