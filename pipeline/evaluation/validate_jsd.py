"""
Accuracy of the Gaussian Jensen-Shannon divergence used for the L2 features.

Reference: adaptive integration (scipy.integrate.quad) of the two expectations in standardized
coordinates, split at breakpoints tied to both scales. Compared against it:
  quad      composite Gauss-Legendre rule (jsd_gaussian_quad)
  table     cubic interpolation of the tabulated function (jsd_gaussian, used by the pipeline)
  gh64      64-point Gauss-Hermite quadrature of each expectation under its own Gaussian
  moment    moment-matched single-Gaussian surrogate
on (a) random standardized parameters, (b) the family of Proposition 2, and (c) flow parameters
sampled from every dataset and L2 feature.

Output: results/tables/jsd_validation.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import integrate

from config import TABLES, RANDOM_STATE
from preprocess import load_dataset, clean
from jsd_gauss import jsd_gaussian, jsd_gaussian_quad, EPS, LN2
from entropy_features import jsd_gaussian_moment_matched

N_SAMPLE = 1500
_OFFS = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 30, 38]


def _hb_term(u, d, r):
    z = (u - d) / r
    lo = min(max(-0.5 * u * u + 0.5 * z * z + np.log(r), -700.0), 700.0)
    hb = np.logaddexp(0.0, -lo) + lo / (1.0 + np.exp(lo))
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi) * hb


def _expect_ref(d, r):
    pts = sorted(set(min(max(p, -9.5), 9.5) for p in
                     [0.0, d] + [s * k for k in _OFFS for s in (-1, 1)] + [d + s * k * r for k in _OFFS for s in (-1, 1)]))
    pts = [-9.5] + [p for p in pts if -9.5 < p < 9.5] + [9.5]
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if b > a:
            tot += integrate.quad(_hb_term, a, b, args=(d, r), limit=200, epsabs=1e-14, epsrel=1e-12)[0]
    return tot


def jsd_ref(m1, s1, m2, s2):
    s1, s2 = max(abs(s1), EPS), max(abs(s2), EPS)
    v = LN2 - 0.5 * (_expect_ref((m2 - m1) / s1, s2 / s1) + _expect_ref((m1 - m2) / s2, s1 / s2))
    return min(max(v, 0.0), LN2)


_GHX, _GHW = np.polynomial.hermite.hermgauss(64)


def jsd_gh64(m1, s1, m2, s2):
    m1, s1, m2, s2 = (np.asarray(v, dtype=float)[:, None] for v in (m1, s1, m2, s2))
    s1, s2 = np.maximum(np.abs(s1), EPS), np.maximum(np.abs(s2), EPS)

    def lp(x, m, s):
        return -0.5 * np.log(2 * np.pi * s * s) - (x - m) ** 2 / (2 * s * s)
    out = 0.0
    for ma, sa in ((m1, s1), (m2, s2)):
        x = ma + np.sqrt(2.0) * sa * _GHX[None, :]
        l1, l2 = lp(x, m1, s1), lp(x, m2, s2)
        own = lp(x, ma, sa)
        out = out + 0.5 * ((own - (np.logaddexp(l1, l2) - LN2)) * _GHW).sum(axis=1) / np.sqrt(np.pi)
    return np.clip(out, 0.0, LN2)


def row(label, m1, s1, m2, s2):
    m1, s1, m2, s2 = (np.asarray(v, dtype=float) for v in (m1, s1, m2, s2))
    ref = np.array([jsd_ref(*t) for t in zip(m1, s1, m2, s2)])
    out = {"set": label, "n": len(ref), "at_bound_share": float((ref > LN2 - 1e-6).mean())}
    raw_mm = jsd_gaussian_moment_matched(m1, np.maximum(np.abs(s1), EPS), m2, np.maximum(np.abs(s2), EPS))
    for name, val in [("quad", jsd_gaussian_quad(m1, s1, m2, s2)), ("table", jsd_gaussian(m1, s1, m2, s2)),
                      ("gh64", jsd_gh64(m1, s1, m2, s2)), ("moment", np.clip(raw_mm, 0.0, LN2))]:
        e = np.abs(val - ref)
        out.update({f"{name}_max_err": e.max(), f"{name}_mean_err": e.mean(),
                    f"{name}_share_gt_1e-3": float((e > 1e-3).mean()), f"{name}_share_gt_1e-2": float((e > 1e-2).mean()),
                    f"{name}_spearman": float(pd.Series(val).corr(pd.Series(ref), method="spearman"))})
    out["moment_share_above_ln2"] = float((raw_mm > LN2).mean())
    print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in out.items()
           if k in ("set", "n", "quad_max_err", "table_max_err", "gh64_max_err", "gh64_share_gt_1e-2", "moment_max_err")}, flush=True)
    return out


def col(df, *names):
    for c in names:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    raise KeyError(names)


rng = np.random.default_rng(RANDOM_STATE)
rows = []

n = 4000
delta = rng.uniform(0.0, 12.0, n)
rho = np.exp(rng.uniform(np.log(1e-9), 0.0, n))
rows.append(row("random standardized", np.zeros(n), np.ones(n), delta, rho))

eps = np.array([1.0, 0.5, 0.2, 0.1, 0.03, 0.01, 1e-3, 1e-5, 1e-8])
rows.append(row("Proposition 2 family", np.full(eps.size, 100.0), np.full(eps.size, 30.0), 100.0 * eps, 30.0 * eps))

for ds in ["CICIDS-2017", "CICIDS-2018", "UNSW-NB15", "NSL-KDD"]:
    df = clean(load_dataset(ds))
    if ds.startswith("CICIDS"):
        feats = {"jsd_pkt_len": (col(df, "Fwd Packet Length Mean", "Fwd Pkt Len Mean"), col(df, "Fwd Packet Length Std", "Fwd Pkt Len Std"),
                                 col(df, "Bwd Packet Length Mean", "Bwd Pkt Len Mean"), col(df, "Bwd Packet Length Std", "Bwd Pkt Len Std")),
                 "jsd_iat": (col(df, "Fwd IAT Mean"), col(df, "Fwd IAT Std"), col(df, "Bwd IAT Mean"), col(df, "Bwd IAT Std"))}
    elif ds == "UNSW-NB15":
        feats = {"jsd_pkt_sz": (df["smeansz"], df["Sjit"], df["dmeansz"], df["Djit"]),
                 "jsd_iat": (df["Sintpkt"], df["Sjit"], df["Dintpkt"], df["Djit"])}
    else:
        a, b = np.log1p(df["src_bytes"].replace(0, EPS)), np.log1p(df["dst_bytes"].replace(0, EPS))
        feats = {"byte_asym_jsd": (a, a * 0.1, b, b * 0.1)}
    for fname, cols in feats.items():
        A = np.column_stack([np.nan_to_num(np.asarray(c, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for c in cols])
        idx = rng.choice(len(A), size=min(N_SAMPLE, len(A)), replace=False)
        rows.append(row(f"{ds} {fname}", *A[idx].T))

out = pd.DataFrame(rows)
TABLES.mkdir(parents=True, exist_ok=True)
out.to_csv(TABLES / "jsd_validation.csv", index=False)
print("\nSaved:", TABLES / "jsd_validation.csv")
print(out[["set", "n", "quad_max_err", "table_max_err", "gh64_max_err", "gh64_share_gt_1e-2", "moment_max_err", "moment_share_above_ln2"]].to_string(index=False))
