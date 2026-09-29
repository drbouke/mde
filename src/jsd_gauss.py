"""
Jensen-Shannon divergence between two univariate Gaussians, in nats.

The divergence depends on the two densities only through the standardized separation
delta = (mu2 - mu1) / s1 and the scale ratio rho = s2 / s1. With w = p / (p + q) and H_b the
binary entropy,

    JSD(P, Q) = ln 2 - 0.5 * E_P[H_b(w)] - 0.5 * E_Q[H_b(w)],

and each expectation is an integral against a standard normal in the coordinates of its own
Gaussian.

jsd_gaussian_quad  evaluates the two integrals by composite 16-point Gauss-Legendre quadrature on
                   panels whose breakpoints follow the scales of both Gaussians.
jsd_gaussian       interpolates a table of the same function on a (ln rho, delta) grid with a
                   cubic spline; the table is built with jsd_gaussian_quad and cached next to
                   this file (jsd_table.npz).

Standard deviations are floored at EPS, so a direction without variation behaves as a point mass:
its divergence from a spread-out Gaussian is ln 2, and two point masses give 0 when they coincide
and ln 2 otherwise.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

EPS = 1e-9
LN2 = float(np.log(2.0))

_GLX, _GLW = np.polynomial.legendre.leggauss(16)
_UMAX = 9.0
_OWN = np.array([-9.0, -6.5, -4.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 9.0])
_OFF = np.array([0.5, 1.0, 2.0, 3.0, 4.5, 6.5, 9.0, 12.0, 16.0, 24.0, 38.0])
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _expect_hb(delta, rho, chunk=20_000):
    """E[H_b(w(U))] for U ~ N(0, 1), where the other Gaussian has mean delta and standard
    deviation rho in the coordinates of the first."""
    delta = np.asarray(delta, dtype=float).ravel()
    rho = np.asarray(rho, dtype=float).ravel()
    out = np.empty(delta.size)
    for a in range(0, delta.size, chunk):
        d = delta[a:a + chunk, None]
        r = rho[a:a + chunk, None]
        other = np.concatenate([d - _OFF[::-1] * r, d, d + _OFF * r], axis=1)
        own = np.broadcast_to(_OWN, (d.shape[0], _OWN.size))
        bp = np.sort(np.clip(np.concatenate([own, other], axis=1), -_UMAX, _UMAX), axis=1)
        half = 0.5 * (bp[:, 1:] - bp[:, :-1])[:, :, None]
        mid = 0.5 * (bp[:, 1:] + bp[:, :-1])[:, :, None]
        u = mid + half * _GLX
        z = (u - d[:, :, None]) / r[:, :, None]
        log_odds = np.clip(-0.5 * u * u + 0.5 * z * z + np.log(r)[:, :, None], -700.0, 700.0)
        hb = np.logaddexp(0.0, -log_odds) + log_odds / (1.0 + np.exp(log_odds))
        out[a:a + chunk] = (np.exp(-0.5 * u * u) / _SQRT2PI * hb * _GLW * half).sum(axis=(1, 2))
    return out


def _prepare(mu1, s1, mu2, s2):
    mu1, s1, mu2, s2 = (np.asarray(v, dtype=float) for v in (mu1, s1, mu2, s2))
    mu1, s1, mu2, s2 = np.broadcast_arrays(mu1, s1, mu2, s2)
    return mu1, np.maximum(np.abs(s1), EPS), mu2, np.maximum(np.abs(s2), EPS)


def jsd_gaussian_quad(mu1, s1, mu2, s2):
    """JSD of N(mu1, s1^2) and N(mu2, s2^2) by direct composite quadrature."""
    mu1, s1, mu2, s2 = _prepare(mu1, s1, mu2, s2)
    e_p = _expect_hb((mu2 - mu1) / s1, s2 / s1)
    e_q = _expect_hb((mu1 - mu2) / s2, s1 / s2)
    return np.clip(LN2 - 0.5 * (e_p + e_q), 0.0, LN2).reshape(mu1.shape)


# ── Tabulated form ───────────────────────────────────────────────────────────────────────
# Grid over a = ln(rho) and delta, with rho = s_min / s_max <= 1 and delta = |mu2 - mu1| / s_max.
# The grid extends slightly past a = 0 and delta = 0 so that every query is interior.
A_MIN, D_MAX = float(np.log(1e-9)), 13.0          # beyond these the divergence is ln 2 to 1e-9
_A_LO, _A_HI, _DA = A_MIN - 0.2, 0.2, 0.04
_D_LO, _D_HI, _DD = -0.2, D_MAX + 0.2, 0.02
_TABLE = Path(__file__).with_name("jsd_table.npz")
_coeff = None


def build_table(path=_TABLE):
    a = np.arange(_A_LO, _A_HI + 0.5 * _DA, _DA)
    d = np.arange(_D_LO, _D_HI + 0.5 * _DD, _DD)
    aa, dd = np.meshgrid(a, d, indexing="ij")
    values = jsd_gaussian_quad(0.0, 1.0, np.abs(dd), np.exp(aa))
    coeff = ndimage.spline_filter(values, order=3, mode="nearest")
    np.savez_compressed(path, coeff=coeff, a_lo=_A_LO, da=_DA, d_lo=_D_LO, dd=_DD)
    return coeff


def _load():
    global _coeff
    if _coeff is None:
        if _TABLE.exists():
            z = np.load(_TABLE)
            ok = (abs(float(z["a_lo"]) - _A_LO) < 1e-12 and abs(float(z["da"]) - _DA) < 1e-12
                  and abs(float(z["d_lo"]) - _D_LO) < 1e-12 and abs(float(z["dd"]) - _DD) < 1e-12)
            _coeff = z["coeff"] if ok else build_table()
        else:
            _coeff = build_table()
    return _coeff


def jsd_gaussian(mu1, s1, mu2, s2):
    """JSD of N(mu1, s1^2) and N(mu2, s2^2) from the tabulated function (cubic interpolation)."""
    mu1, s1, mu2, s2 = _prepare(mu1, s1, mu2, s2)
    big, small = np.maximum(s1, s2), np.minimum(s1, s2)
    a = np.log(small / big).ravel()
    d = (np.abs(mu2 - mu1) / big).ravel()
    out = np.full(a.shape, LN2)
    inside = (a > A_MIN) & (d < D_MAX)
    if inside.any():
        coords = np.vstack([(a[inside] - _A_LO) / _DA, (d[inside] - _D_LO) / _DD])
        out[inside] = ndimage.map_coordinates(_load(), coords, order=3, mode="nearest", prefilter=False)
    return np.clip(out, 0.0, LN2).reshape(mu1.shape)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    c = build_table()
    print(f"table {c.shape} written to {_TABLE.name} in {time.perf_counter() - t0:.1f} s")
