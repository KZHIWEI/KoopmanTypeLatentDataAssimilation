from __future__ import annotations

import numpy as np

__all__ = ["gram_matrices", "residuals", "summary", "self_test"]


def gram_matrices(X: np.ndarray, Y: np.ndarray):
    M = X.shape[0]
    G = X.conj().T @ X / M
    A = X.conj().T @ Y / M
    L = Y.conj().T @ Y / M
    return G, A, L


def residuals(
    X: np.ndarray, Y: np.ndarray, rcond: float = 1e-6, gram_floor: float = 1e-8
):
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    r = int((sv > sv[0] * rcond).sum())
    n_dropped = int(X.shape[1] - r)
    P = Vt[:r].T
    Xr, Yr = X @ P, Y @ P

    G, A, L = gram_matrices(Xr, Yr)
    K = np.linalg.solve(G + 1e-14 * np.trace(G) / r * np.eye(r), A)
    lam, V = np.linalg.eig(K)
    X = Xr

    def _diag_form(M):
        return (V.conj() * (M @ V)).sum(axis=0)

    gLg, gAg, gGg = _diag_form(L), _diag_form(A), _diag_form(G)
    gAhg = _diag_form(A.conj().T)
    num = gLg - lam * gAhg - np.conj(lam) * gAg + np.abs(lam) ** 2 * gGg
    res = np.sqrt(np.maximum(np.real(num / gGg), 0.0))
    Phi = X @ V
    weight = np.real(np.einsum("ij,ij->j", Phi.conj(), Phi))
    weight = weight / max(weight.sum(), 1e-30)
    return res, lam, weight, n_dropped


def residuals_given(lam, V, X, Y, gram_floor: float = 1e-8):
    lam = np.asarray(lam, dtype=np.complex128)
    V = np.asarray(V, dtype=np.complex128)
    X = np.asarray(X, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    G, A, L = gram_matrices(X, Y)

    def _diag_form(M):
        return (V.conj() * (M @ V)).sum(axis=0)

    gGg = _diag_form(G)
    gGg_r = np.real(gGg)
    kept = gGg_r > gram_floor * max(gGg_r.max(), 1e-300)

    num = (
        _diag_form(L)
        - lam * _diag_form(A.conj().T)
        - np.conj(lam) * _diag_form(A)
        + np.abs(lam) ** 2 * gGg
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.sqrt(np.maximum(np.real(num / gGg), 0.0))
    res = np.where(kept, res, np.nan)

    Phi = X @ V
    weight = np.real(np.einsum("ij,ij->j", Phi.conj(), Phi)) * kept
    weight = weight / max(weight.sum(), 1e-30)
    return res, weight, kept


def summary(X: np.ndarray, Y: np.ndarray, tol: float = 0.1) -> dict:
    res, lam, w, n_dropped = residuals(X, Y)
    order = np.argsort(res)
    return {
        "n_eigenpairs": int(len(res)),
        "n_dropped_below_gram_floor": n_dropped,
        "cond_snapshots": None,
        "res_median": float(np.median(res)),
        "res_mean": float(res.mean()),
        "res_energy_weighted": float((res * w).sum()),
        "res_p10": float(np.percentile(res, 10)),
        "res_p90": float(np.percentile(res, 90)),
        "res_min": float(res.min()),
        f"frac_below_{tol}": float((res < tol).mean()),
        "best_eigenvalues": [
            {
                "lambda_re": float(np.real(lam[j])),
                "lambda_im": float(np.imag(lam[j])),
                "res": float(res[j]),
            }
            for j in order[:5]
        ],
    }


def self_test(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    M, D = 4000, 24
    X = rng.normal(size=(M, D))
    K_true = rng.normal(size=(D, D)) / np.sqrt(D)

    lin = X @ K_true
    nl = lin + 0.30 * np.tanh(X @ K_true)

    r_lin, *_ = residuals(X, lin)
    r_nl, *_ = residuals(X, nl)

    S = np.diag(np.exp(rng.uniform(-3.5, 3.5, D)))
    r_lin_s, *_ = residuals(X @ S, lin @ S)
    r_nl_s, *_ = residuals(X @ S, nl @ S)

    def drift(a, b):
        a, b = np.sort(a), np.sort(b)
        absd = float(np.max(np.abs(a - b)))
        base = float(np.max(np.abs(a)))
        return {
            "abs": absd,
            "rel": (absd / base if base > 1e-6 else None),
            "base": base,
            "rel_meaningful": bool(base > 1e-6),
        }

    W = rng.normal(size=(D, D)) / np.sqrt(D)
    assert np.abs(W - W.T).max() > 0.1
    Yw = X @ W
    lam_w, V_w = np.linalg.eig(W.astype(np.complex128))
    r_own, _, kept_own = residuals_given(lam_w, V_w, X, Yw)
    lam_t, V_t = np.linalg.eig(W.T.astype(np.complex128))
    r_wrong, _, _ = residuals_given(lam_t, V_t, X, Yw)

    return {
        "linear_res_max": float(r_lin.max()),
        "nonlinear_res_median": float(np.median(r_nl)),
        "rescale_drift_linear": drift(r_lin, r_lin_s),
        "rescale_drift_nonlinear": drift(r_nl, r_nl_s),
        "own_operator_res_max": float(np.nanmax(r_own)),
        "own_operator_n_kept": int(kept_own.sum()),
        "wrong_transpose_res_median": float(np.nanmedian(r_wrong)),
        "verdict": {
            "vanishes_on_linear": bool(r_lin.max() < 1e-6),
            "nonzero_on_nonlinear": bool(np.median(r_nl) > 1e-3),
            "scale_invariant": bool(drift(r_nl, r_nl_s)["rel"] < 1e-6),
            "vanishes_on_own_operator": bool(np.nanmax(r_own) < 1e-6),
            "wrong_transpose_is_caught": bool(np.nanmedian(r_wrong) > 0.1),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=1))
