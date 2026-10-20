from __future__ import annotations

from typing import Dict, List

import numpy as np
import scipy.stats as sst

from src.eval.gaussianity import stats as S

SAMPLES_PER_DIM = 20


def _k_for(n: int, D: int, k_req: int) -> int:
    return int(max(2, min(k_req, D, n // SAMPLES_PER_DIM)))


def skewness_is_estimable(n: int, k: int) -> bool:
    return n >= 2 * (k * (k + 1) * (k + 2) / 6)


def _panel(E: np.ndarray, U: np.ndarray) -> Dict:
    Z = S.project(E, U)
    lil = [S.lilliefors_test(Z[:, j]) for j in range(Z.shape[1])]
    return {
        "hz": S.henze_zirkler_test(Z),
        "mardia_kurtosis": S.mardia_kurtosis_test(Z),
        "mardia_skewness": S.mardia_skewness_test(Z),
        "lilliefors_worst_coord": S.bonferroni_min([t["p"] for t in lil]),
        "lilliefors_per_coord_p": [t["p"] for t in lil],
        "lilliefors_p_at_floor": int(sum(t["p_at_floor"] for t in lil)),
        "excess_kurtosis_pooled": float(
            sst.kurtosis(((Z - Z.mean(0)) / (Z.std(0) + 1e-300)).ravel())
        ),
        "coord_std": [float(v) for v in Z.std(0)],
        "k": int(Z.shape[1]),
    }


def _whitened_panel(E: np.ndarray, kind: str, k: int, seed: int) -> Dict:
    Z, meta = S.whiten_empirical(E) if kind == "emp" else S.whiten_ledoit_wolf(E)
    U = S.basis_random(Z.shape[1], _k_for(Z.shape[0], Z.shape[1], k), seed)
    Y = Z @ U
    lil = [S.lilliefors_test(Y[:, j]) for j in range(Y.shape[1])]
    return {
        "whitening": meta,
        "lilliefors_worst_coord": S.bonferroni_min([t["p"] for t in lil]),
        "excess_kurtosis_pooled": float(sst.kurtosis(Z.ravel())),
        "ks_to_normal_legacy": float(
            sst.kstest(Z.ravel() / (Z.ravel().std() + 1e-30), "norm").statistic
        ),
        "k": int(Y.shape[1]),
    }


def per_start(
    E_b: np.ndarray, E_a: np.ndarray, B: np.ndarray, *, k: int = 8, seed: int = 20260802
) -> Dict:
    n, D = E_a.shape
    kk = _k_for(n, D, k)
    U_b = S.basis_from_cov(B, kk)
    U_m = S.basis_from_cov_min(B, kk)
    U_r = S.basis_random(D, kk, seed)
    out = {"n": int(n), "D": int(D), "k": int(kk)}
    for tag, E in (("background", E_b), ("analysis", E_a)):
        out[tag] = {
            "descriptive": S.descriptive(E),
            "proj_B": _panel(E, U_b),
            "proj_Bmin": _panel(E, U_m),
            "proj_rand": _panel(E, U_r),
            "lw": _whitened_panel(E, "lw", k, seed),
            "emp": _whitened_panel(E, "emp", k, seed),
        }
    db = out["background"]["descriptive"]
    da = out["analysis"]["descriptive"]
    out["trace_ratio"] = float(da["trace"] / max(db["trace"], 1e-300))
    out["rms_ratio"] = float(da["rms"] / max(db["rms"], 1e-300))
    return out


TEST_PATHS = {
    "hz_projB": ("proj_B", "hz", "p"),
    "hz_projBmin": ("proj_Bmin", "hz", "p"),
    "hz_projrand": ("proj_rand", "hz", "p"),
    "mardia_projB": ("proj_B", "mardia_kurtosis", "p"),
    "mardia_skew_projB": ("proj_B", "mardia_skewness", "p"),
    "lilliefors_projB": ("proj_B", "lilliefors_worst_coord", "p"),
    "lilliefors_lw": ("lw", "lilliefors_worst_coord", "p"),
    "lilliefors_emp": ("emp", "lilliefors_worst_coord", "p"),
}


def combine(starts: List[Dict], tag: str = "analysis") -> Dict:
    out = {}
    n, k = int(starts[0]["n"]), int(starts[0]["k"])
    for name, path in TEST_PATHS.items():
        if "mardia_skew" in name and not skewness_is_estimable(n, k):
            out[name] = {"skipped": True, "df": k * (k + 1) * (k + 2) // 6, "n": n}
            continue
        ps = []
        for s in starts:
            v = s[tag]
            for key in path:
                v = v[key]
            ps.append(v)
        out[name] = S.fisher_combine(ps)
        out[name]["per_start_p"] = [None if p is None else float(p) for p in ps]
    return out


def summarise(starts: List[Dict]) -> Dict:
    med = lambda path: float(np.median([_dig(s, path) for s in starts]))
    return {
        "n_starts": len(starts),
        "analysis": combine(starts, "analysis"),
        "background": combine(starts, "background"),
        "median": {
            "trace_ratio": med(("trace_ratio",)),
            "rms_background": med(("background", "descriptive", "rms")),
            "rms_analysis": med(("analysis", "descriptive", "rms")),
            "trace_background": med(("background", "descriptive", "trace")),
            "trace_analysis": med(("analysis", "descriptive", "trace")),
            "mean_bias_analysis": med(("analysis", "descriptive", "mean_bias")),
            "excess_kurtosis_analysis_emp": med(
                ("analysis", "emp", "excess_kurtosis_pooled")
            ),
            "excess_kurtosis_analysis_lw": med(
                ("analysis", "lw", "excess_kurtosis_pooled")
            ),
            "excess_kurtosis_analysis_projB": med(
                ("analysis", "proj_B", "excess_kurtosis_pooled")
            ),
            "ks_analysis_emp_legacy": med(("analysis", "emp", "ks_to_normal_legacy")),
            "mardia_z_analysis_projB": med(
                ("analysis", "proj_B", "mardia_kurtosis", "z")
            ),
            "excess_kurtosis_background_emp": med(
                ("background", "emp", "excess_kurtosis_pooled")
            ),
            "ks_background_emp_legacy": med(
                ("background", "emp", "ks_to_normal_legacy")
            ),
            "mardia_z_background_projB": med(
                ("background", "proj_B", "mardia_kurtosis", "z")
            ),
            "mardia_skew_z_background_projB": med(
                ("background", "proj_B", "mardia_skewness", "z")
            ),
            "mardia_skew_z_analysis_projB": med(
                ("analysis", "proj_B", "mardia_skewness", "z")
            ),
        },
    }


def _dig(d: Dict, path):
    v = d
    for k in path:
        v = v[k]
    return v
