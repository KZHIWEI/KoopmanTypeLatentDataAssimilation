from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.stats as sst

LILLIEFORS_P_FLOOR = 1e-3


def basis_from_cov(B: np.ndarray, k: int) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(B, dtype=np.float64))
    order = np.argsort(w)[::-1][:k]
    return np.ascontiguousarray(V[:, order])


def basis_from_cov_min(B: np.ndarray, k: int) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(B, dtype=np.float64))
    order = np.argsort(w)[:k]
    return np.ascontiguousarray(V[:, order])


def basis_random(D: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return np.ascontiguousarray(Q[:, :k])


def project(E: np.ndarray, U: np.ndarray) -> np.ndarray:
    return np.asarray(E, dtype=np.float64) @ np.asarray(U, dtype=np.float64)


def whiten_empirical(E: np.ndarray, rtol: float = 1e-8) -> Tuple[np.ndarray, Dict]:
    E = np.asarray(E, dtype=np.float64)
    n = E.shape[0]
    Ec = E - E.mean(0)
    C = (Ec.T @ Ec) / max(1, n - 1)
    w, V = np.linalg.eigh(C)
    keep = w > max(w.max() * rtol, 1e-300)
    Z = Ec @ (V[:, keep] / np.sqrt(w[keep]))
    return Z, {
        "kind": "empirical",
        "rank": int(keep.sum()),
        "n": n,
        "D": int(E.shape[1]),
    }


def whiten_ledoit_wolf(E: np.ndarray) -> Tuple[np.ndarray, Dict]:
    from sklearn.covariance import LedoitWolf

    E = np.asarray(E, dtype=np.float64)
    lw = LedoitWolf(assume_centered=False, store_precision=False).fit(E)
    C = lw.covariance_
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, max(w.max(), 1e-300) * 1e-12, None)
    Z = (E - lw.location_) @ (V / np.sqrt(w)) @ V.T
    return Z, {
        "kind": "ledoit_wolf",
        "shrinkage": float(lw.shrinkage_),
        "n": int(E.shape[0]),
        "D": int(E.shape[1]),
    }


def lilliefors_test(x: np.ndarray) -> Dict:
    from statsmodels.stats.diagnostic import lilliefors

    x = np.asarray(x, dtype=np.float64).ravel()
    stat, p = lilliefors(x, dist="norm", pvalmethod="approx")
    return {
        "stat": float(stat),
        "p": float(p),
        "n": int(x.size),
        "p_at_floor": bool(float(p) <= LILLIEFORS_P_FLOOR),
    }


def henze_zirkler_test(Z: np.ndarray) -> Dict:
    import pingouin as pg

    Z = np.asarray(Z, dtype=np.float64)
    res = pg.multivariate_normality(Z, alpha=0.05)
    return {
        "stat": float(res.hz),
        "p": float(res.pval),
        "n": int(Z.shape[0]),
        "k": int(Z.shape[1]),
    }


def mardia_skewness_test(Z: np.ndarray, ridge: float = 1e-10) -> Dict:
    Z = np.asarray(Z, dtype=np.float64)
    n, p = Z.shape
    Zc = Z - Z.mean(0)
    S = (Zc.T @ Zc) / n
    S = S + ridge * np.trace(S) / p * np.eye(p)
    M = Zc @ np.linalg.inv(S) @ Zc.T
    b1p = float((M**3).mean())
    df = p * (p + 1) * (p + 2) / 6.0
    A = n * b1p / 6.0
    z = (A - df) / np.sqrt(2.0 * df)
    return {
        "b1p": b1p,
        "A": A,
        "df": float(df),
        "z": float(z),
        "p": float(sst.chi2.sf(A, df)),
        "n": int(n),
        "k": int(p),
        "one_sided": True,
    }


def mardia_kurtosis_test(Z: np.ndarray, ridge: float = 1e-10) -> Dict:
    Z = np.asarray(Z, dtype=np.float64)
    n, p = Z.shape
    Zc = Z - Z.mean(0)
    S = (Zc.T @ Zc) / n
    S = S + ridge * np.trace(S) / p * np.eye(p)
    d2 = np.einsum("ij,jk,ik->i", Zc, np.linalg.inv(S), Zc)
    b2p = float((d2**2).mean())
    mean, var = p * (p + 2), 8.0 * p * (p + 2) / n
    z = (b2p - mean) / np.sqrt(var)
    return {
        "b2p": b2p,
        "expected": float(mean),
        "z": float(z),
        "p": float(2.0 * sst.norm.sf(abs(z))),
        "n": int(n),
        "k": int(p),
    }


def fisher_combine(ps: Sequence[float]) -> Dict:
    ps = [
        float(min(max(p, 1e-300), 1.0)) for p in ps if p is not None and np.isfinite(p)
    ]
    if not ps:
        return {"p": None, "stat": None, "m": 0}
    stat, p = sst.combine_pvalues(ps, method="fisher")
    return {"p": float(p), "stat": float(stat), "m": len(ps)}


def bonferroni_min(ps: Sequence[float]) -> Dict:
    ps = [float(p) for p in ps if p is not None and np.isfinite(p)]
    if not ps:
        return {"p": None, "m": 0}
    return {
        "p": float(min(1.0, min(ps) * len(ps))),
        "m": len(ps),
        "p_min": float(min(ps)),
    }


def descriptive(E: np.ndarray) -> Dict:
    E = np.asarray(E, dtype=np.float64)
    n = E.shape[0]
    mu = E.mean(0)
    Ec = E - mu
    C = (Ec.T @ Ec) / max(1, n - 1)
    return {
        "rms": float(np.sqrt((E**2).sum(1).mean())),
        "trace": float(np.trace(C)),
        "mean_bias": float(np.linalg.norm(mu)),
        "n": int(n),
        "D": int(E.shape[1]),
    }
