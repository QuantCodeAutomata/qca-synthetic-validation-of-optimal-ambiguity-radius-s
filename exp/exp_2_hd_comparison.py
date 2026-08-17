"""
Experiment 2: Synthetic high-dimensional covariance estimation error comparison.

Compares sample, LW-L, LW-NL, W-DRCOE, WISE, W-SCOPE for p in {150, 200}
over a range of sample sizes, using 50 Monte Carlo repetitions.

Oracle tuning of rho for W-SCOPE, W-DRCOE, WISE: grid search over 100 log-spaced
rho values in [1e-5, 2e3], selecting the rho that minimises the average oracle
loss over all 50 repetitions.

Saves:
  results/exp2_combined_loss_table.csv
  results/exp2_eigenvalue_error_table.csv
"""

import sys
sys.path.insert(0, '/workspace/project')

import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

from sklearn.covariance import LedoitWolf  # for LW-L shrinkage coefficient

from src.scope import (
    make_true_covariance,
    wasserstein_div_v,
    wasserstein_phi_batch,
    # Fast eigenvalue-only variants (skip V @ diag @ V.T reconstruction)
    oracle_combined_loss_from_eig,
)
from src.benchmark_estimators import (
    sample_cov,
    lw_nonlinear_shrinkage_eig,  # vectorised LW-NL on precomputed eigenvalues
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

METHODS = ['sample', 'LW-L', 'LW-NL', 'W-DRCOE', 'WISE', 'W-SCOPE']

N_REPS = 50
N_RHO = 100
RHO_GRID = np.logspace(-5, np.log10(2e3), N_RHO)

# Methods that require oracle rho tuning and which oracle loss to minimise
TUNE_LOSS = {
    'W-SCOPE':  'combined',
    'W-DRCOE':  'frobenius',
    'WISE':     'stein',
}

P_N_CONFIG = {
    150: [50, 70, 100, 120, 150, 200, 300],
    200: [50, 100, 120, 150, 170, 200, 300, 400],
}

# ---------------------------------------------------------------------------
# Fast vectorised helpers for tuning (avoid per-eigenvalue brentq calls)
# ---------------------------------------------------------------------------

def _w_phi_bisect(tau: float, gamma: float, lambdas: np.ndarray,
                  n_iter: int = 64) -> np.ndarray:
    """W-SCOPE phi via fully vectorised bisection across all eigenvalues.

    Solves  tau*a^2 + gamma*a - gamma*sqrt(b)*sqrt(a) - 1 = 0  simultaneously
    for all b in `lambdas`.  Replaces p individual brentq calls with a single
    batch bisection loop, giving a ~100x speed-up for the rho grid search.
    """
    sqrt_lam = np.sqrt(lambdas)

    def f(a: np.ndarray) -> np.ndarray:
        return tau * a ** 2 + gamma * a - gamma * sqrt_lam * np.sqrt(np.maximum(a, 1e-20)) - 1.0

    lo = np.full(len(lambdas), 1e-15)
    hi = np.maximum(1.0 / np.sqrt(tau), lambdas) * 10.0 + 10.0

    # Ensure f(hi) > 0 everywhere
    f_hi = f(hi)
    bad = f_hi <= 0
    while np.any(bad):
        hi[bad] = hi[bad] * 100.0 + 1e6
        f_hi = np.where(bad, f(hi), f_hi)
        bad = f_hi <= 0

    f_lo = f(lo)  # always negative (= -1 at a→0)

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        move_lo = f_lo * f_mid > 0
        lo = np.where(move_lo, mid, lo)
        f_lo = np.where(move_lo, f_mid, f_lo)
        hi = np.where(~move_lo, mid, hi)

    return np.maximum((lo + hi) / 2.0, 1e-12)


def _w_scope_gamma_residual(gamma: float, tau: float, rho: float,
                             lambdas: np.ndarray) -> float:
    """Dual residual  sum_i d_W(phi_i, lambda_i) - rho  for W-SCOPE."""
    phi = _w_phi_bisect(tau, gamma, lambdas)
    return float(np.sum(wasserstein_div_v(phi, lambdas))) - rho


def _scope_w_eigenvalues(lambdas: np.ndarray, tau: float,
                          rho: float) -> np.ndarray:
    """W-SCOPE shrunk eigenvalues using the fast vectorised phi."""
    scalar_target = 1.0 / np.sqrt(tau)
    rho_max = float(np.sum(wasserstein_div_v(scalar_target, lambdas)))

    if rho >= rho_max:
        return np.full_like(lambdas, scalar_target)

    gamma_hi = 1.0
    while _w_scope_gamma_residual(gamma_hi, tau, rho, lambdas) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f0 = _w_scope_gamma_residual(0.0, tau, rho, lambdas)
    f_hi = _w_scope_gamma_residual(gamma_hi, tau, rho, lambdas)

    if f0 <= 0:
        return np.full_like(lambdas, scalar_target)
    if f_hi >= 0:
        return np.maximum(_w_phi_bisect(tau, gamma_hi, lambdas), 1e-12)

    gamma_star = brentq(
        _w_scope_gamma_residual, 0.0, gamma_hi,
        args=(tau, rho, lambdas), xtol=1e-10, rtol=1e-10,
    )
    return np.maximum(_w_phi_bisect(tau, gamma_star, lambdas), 1e-12)


def _wdrcoe_eigs_v(gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """W-DRCOE eigenvalues given Lagrange multiplier gamma (vectorised)."""
    s = ((np.sqrt(lambdas) + np.sqrt(lambdas + 4.0 / gamma)) / 2.0) ** 2
    return np.maximum(s, 1e-12)


def _wdrcoe_gamma_residual(gamma: float, lambdas: np.ndarray, rho: float) -> float:
    s = _wdrcoe_eigs_v(gamma, lambdas)
    return float(np.sum(wasserstein_div_v(s, lambdas))) - rho


def _wdrcoe_eigenvalues(lambdas: np.ndarray, rho: float) -> np.ndarray:
    """W-DRCOE shrunk eigenvalues (vectorised, fast)."""
    if rho <= 1e-14:
        return lambdas.copy()

    gamma_hi = 1.0
    while _wdrcoe_gamma_residual(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wdrcoe_gamma_residual(1e-10, lambdas, rho)
    f_hi = _wdrcoe_gamma_residual(gamma_hi, lambdas, rho)

    if f_lo < 0:
        return lambdas.copy()
    if f_hi > 0:
        return _wdrcoe_eigs_v(gamma_hi, lambdas)

    gamma_star = brentq(
        _wdrcoe_gamma_residual, 1e-10, gamma_hi,
        args=(lambdas, rho), xtol=1e-10, rtol=1e-10,
    )
    return _wdrcoe_eigs_v(gamma_star, lambdas)


def _wise_eigs_v(gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """WISE eigenvalues given Lagrange multiplier gamma (vectorised)."""
    gb = gamma * lambdas
    v = (-np.sqrt(gb) + np.sqrt(gb + 4.0)) / 2.0
    mu = np.maximum(v ** 2, 1e-12)
    return np.minimum(1.0 / mu, 1e12)


def _wise_gamma_residual(gamma: float, lambdas: np.ndarray, rho: float) -> float:
    s = _wise_eigs_v(gamma, lambdas)
    return float(np.sum(wasserstein_div_v(s, lambdas))) - rho


def _wise_eigenvalues(lambdas: np.ndarray, rho: float) -> np.ndarray:
    """WISE shrunk eigenvalues (vectorised, fast)."""
    if rho <= 1e-14:
        return lambdas.copy()

    gamma_hi = 1.0
    while _wise_gamma_residual(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wise_gamma_residual(1e-10, lambdas, rho)
    f_hi = _wise_gamma_residual(gamma_hi, lambdas, rho)

    if f_lo < 0:
        return lambdas.copy()
    if f_hi > 0:
        return _wise_eigs_v(gamma_hi, lambdas)

    gamma_star = brentq(
        _wise_gamma_residual, 1e-10, gamma_hi,
        args=(lambdas, rho), xtol=1e-10, rtol=1e-10,
    )
    return _wise_eigs_v(gamma_star, lambdas)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n: int, p: int, Sigma_0: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw n zero-mean Gaussian samples with covariance Sigma_0."""
    L = np.linalg.cholesky(Sigma_0)
    Z = rng.standard_normal((n, p))
    return Z @ L.T


# ---------------------------------------------------------------------------
# Fast O(p) oracle loss formulas from eigenvalues
# ---------------------------------------------------------------------------

def _combined_loss_fast(phi: np.ndarray, M_diag: np.ndarray,
                         tau_star: float) -> float:
    """Oracle combined Stein-Frobenius loss from eigenvalues, O(p).

    Requires Sigma_est = V diag(phi) V^T  and  M_diag = diag(V^T Sigma_0 V).

    Loss = sum log(phi_i) + sum M_diag_i/phi_i
           + (tau*/2)*(sum phi_i^2 - 2*sum phi_i*M_diag_i)
    """
    if np.any(phi <= 0):
        return np.inf
    log_phi = np.log(phi)
    if not np.all(np.isfinite(log_phi)):
        return np.inf
    stein = np.sum(log_phi) + np.sum(M_diag / phi)
    frob = (tau_star / 2.0) * (np.sum(phi ** 2) - 2.0 * np.sum(phi * M_diag))
    v = stein + frob
    return float(v) if np.isfinite(v) else np.inf


def _frobenius_loss_fast(phi: np.ndarray, M_diag: np.ndarray,
                          sigma0_frob2: float) -> float:
    """Oracle Frobenius loss ||Sigma_est - Sigma_0||_F^2 from eigenvalues, O(p)."""
    v = np.sum(phi ** 2) - 2.0 * np.sum(phi * M_diag) + sigma0_frob2
    return float(v) if np.isfinite(v) else np.inf


def _stein_loss_fast(phi: np.ndarray, M_diag: np.ndarray) -> float:
    """Oracle Stein loss log det Sigma_est + tr(Sigma_est^{-1} Sigma_0), O(p)."""
    if np.any(phi <= 0):
        return np.inf
    log_phi = np.log(phi)
    if not np.all(np.isfinite(log_phi)):
        return np.inf
    v = np.sum(log_phi) + np.sum(M_diag / phi)
    return float(v) if np.isfinite(v) else np.inf


# ---------------------------------------------------------------------------
# Oracle rho tuning helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batch helpers for gamma-parametric oracle tuning  (no Python loops)
# ---------------------------------------------------------------------------

# Shared gamma grid: 400 log-spaced values in [1e-6, 1e9].
# All three methods use gamma-parametric curve tracing, which replaces
# N_RHO × N_REPS brentq calls with N_REPS vectorised (K, p) array ops.
_GAMMA_GRID = np.logspace(-6, 9, 400)


def _wdrcoe_eigs_batch(gammas: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """W-DRCOE eigenvalues for a (K,) gamma grid and (p,) lambdas → (K, p).

    Closed form: s_i = ((sqrt(b_i) + sqrt(b_i + 4/gamma)) / 2)^2.
    As gamma → ∞, s_i → b_i (no shrinkage, rho → 0).
    As gamma → 0, s_i → ∞ (over-expansion, rho → ∞).
    """
    g = np.maximum(gammas[:, None], 1e-30)       # (K, 1)
    b = lams[None, :]                             # (1, p)
    return np.maximum(((np.sqrt(b) + np.sqrt(b + 4.0 / g)) / 2.0) ** 2, 1e-12)


def _wise_eigs_batch(gammas: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """WISE eigenvalues for a (K,) gamma grid and (p,) lambdas → (K, p).

    Closed form: mu_i = v_i^2, where v_i = (-sqrt(gamma*b_i) + sqrt(gamma*b_i+4))/2;
    return precision eigenvalues 1/mu_i clipped to avoid overflow.
    """
    g = gammas[:, None]                           # (K, 1)
    b = lams[None, :]                             # (1, p)
    gb = g * b
    v = (-np.sqrt(np.maximum(gb, 0.0)) + np.sqrt(gb + 4.0)) / 2.0
    mu = np.maximum(v ** 2, 1e-12)
    return np.minimum(1.0 / mu, 1e12)            # WISE gives precision eigs; invert → cov eigs


def _w_div_batch(phi: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Wasserstein divergence sum for (K, p) phi and (p,) lams → (K,)."""
    return np.sum((np.sqrt(phi) - np.sqrt(lams[None, :])) ** 2, axis=1)


def _combined_loss_batch(phi: np.ndarray, M_diag: np.ndarray,
                          tau_star: float) -> np.ndarray:
    """Oracle combined Stein-Frobenius loss for (K, p) phi array → (K,)."""
    phi_safe = np.maximum(phi, 1e-300)
    stein = np.sum(np.log(phi_safe) + M_diag[None, :] / phi_safe, axis=1)
    frob = (tau_star / 2.0) * np.sum(phi ** 2 - 2.0 * phi * M_diag[None, :], axis=1)
    return np.where(np.isfinite(stein + frob), stein + frob, np.inf)


def _frobenius_loss_batch(phi: np.ndarray, M_diag: np.ndarray,
                           sigma0_frob2: float) -> np.ndarray:
    """Oracle Frobenius loss for (K, p) phi array → (K,)."""
    v = (np.sum(phi ** 2, axis=1)
         - 2.0 * np.sum(phi * M_diag[None, :], axis=1)
         + sigma0_frob2)
    return np.where(np.isfinite(v), v, np.inf)


def _stein_loss_batch(phi: np.ndarray, M_diag: np.ndarray) -> np.ndarray:
    """Oracle Stein loss for (K, p) phi array → (K,)."""
    phi_safe = np.maximum(phi, 1e-300)
    v = np.sum(np.log(phi_safe) + M_diag[None, :] / phi_safe, axis=1)
    return np.where(np.isfinite(v), v, np.inf)


def _interp_onto_rho_grid(rho_all: np.ndarray, loss_all: np.ndarray) -> np.ndarray:
    """Sort (rho, loss) points and interpolate loss at each RHO_GRID value."""
    valid = np.isfinite(loss_all) & (rho_all >= 0.0)
    if not np.any(valid):
        return np.full(N_RHO, np.inf)
    ra, la = rho_all[valid], loss_all[valid]
    idx = np.argsort(ra)
    ra, la = ra[idx], la[idx]
    return np.interp(RHO_GRID, ra, la, left=la[0], right=la[-1])


# ---------------------------------------------------------------------------
# Oracle rho tuning for one (p, n) configuration  (fully vectorised)
# ---------------------------------------------------------------------------

def tune_rho(p: int, n: int, Sigma_0: np.ndarray, tau_star: float) -> dict:
    """Find the rho minimising average oracle loss via gamma-parametric tracing.

    All three tuned methods (W-SCOPE, W-DRCOE, WISE) now use a single batch
    (K, p) call per rep instead of nested Python loops, giving ~30-50× speedup:

      W-SCOPE  : wasserstein_phi_batch(tau_hat, _GAMMA_GRID, lams) → (K, p)
      W-DRCOE  : _wdrcoe_eigs_batch(_GAMMA_GRID, lams)             → (K, p)
      WISE     : _wise_eigs_batch(_GAMMA_GRID, lams)               → (K, p)

    For each rep the (rho, loss) curve is traced over 400 gamma values, sorted
    by rho, then interpolated onto RHO_GRID.  Losses are averaged over reps
    and the minimising rho is returned.
    """
    sigma0_frob2 = float(np.sum(Sigma_0 ** 2))

    # ── Precompute eigendecompositions and M_diag once per rep ──────────────
    eigen_cache: list[tuple] = []
    for rep in range(N_REPS):
        rng = np.random.default_rng(p * 10000 + n * 100 + rep)
        X = generate_data(n, p, Sigma_0, rng)
        S = sample_cov(X, zero_mean=True)
        lams, V = np.linalg.eigh(S)
        lams = np.maximum(lams, 1e-12)
        tau_hat = float(p / np.sum(S ** 2))
        M_diag = np.sum((Sigma_0 @ V) * V, axis=0)   # diag(V^T Sigma_0 V)
        eigen_cache.append((lams, V, tau_hat, M_diag))

    tuned_rhos: dict = {}

    for method, _loss_type in TUNE_LOSS.items():
        avg_losses = np.zeros(N_RHO)
        counts = np.zeros(N_RHO)

        for lams, _V, tau_hat, M_diag in eigen_cache:
            # ── Single batch call → (K, p) shrunk eigenvalues ───────────────
            if method == 'W-SCOPE':
                phi = wasserstein_phi_batch(tau_hat, _GAMMA_GRID, lams)   # (K, p)
                scalar_target = 1.0 / np.sqrt(tau_hat)
                rho_max = float(np.sum(wasserstein_div_v(scalar_target, lams)))
                rho_all = _w_div_batch(phi, lams)                         # (K,)
                loss_all = _combined_loss_batch(phi, M_diag, tau_star)    # (K,)
                # Append the scalar-target endpoint at rho = rho_max
                loss_sc = _combined_loss_fast(
                    np.full_like(lams, scalar_target), M_diag, tau_star)
                rho_all = np.append(rho_all, rho_max)
                loss_all = np.append(loss_all, loss_sc)

            elif method == 'W-DRCOE':
                phi = _wdrcoe_eigs_batch(_GAMMA_GRID, lams)               # (K, p)
                rho_all = _w_div_batch(phi, lams)                         # (K,)
                loss_all = _frobenius_loss_batch(phi, M_diag, sigma0_frob2)

            else:  # WISE — phi returns covariance eigenvalues (1/mu)
                phi = _wise_eigs_batch(_GAMMA_GRID, lams)                 # (K, p)
                rho_all = _w_div_batch(phi, lams)                         # (K,)
                loss_all = _stein_loss_batch(phi, M_diag)

            rep_losses = _interp_onto_rho_grid(rho_all, loss_all)        # (N_RHO,)
            fm = np.isfinite(rep_losses)
            avg_losses[fm] += rep_losses[fm]
            counts[fm] += 1

        valid = counts > 0
        avg_losses[valid] /= counts[valid]
        avg_losses[~valid] = np.inf

        best_ri = int(np.argmin(avg_losses))
        tuned_rhos[method] = RHO_GRID[best_ri]
        print(f"      {method}: best rho = {tuned_rhos[method]:.4g}  "
              f"(avg loss = {avg_losses[best_ri]:.4g})")

    # Return eigen_cache so evaluate() can reuse eigendecompositions instead
    # of re-generating data and re-running eigh for each of the 6 methods.
    return tuned_rhos, eigen_cache


# ---------------------------------------------------------------------------
# Final evaluation for one (p, n) configuration
# ---------------------------------------------------------------------------

def evaluate(
    p: int,
    n: int,
    Sigma_0: np.ndarray,
    tau_star: float,
    tuned_rhos: dict,
    eigen_cache: list,
) -> tuple:
    """Compute average combined loss and eigenvalue error over 50 reps.

    Fast path: uses ``eigen_cache`` produced by ``tune_rho()`` so that
    eigendecompositions and M_diag are not recomputed.  X is generated
    once per rep (only for LW-L shrinkage coefficient) instead of once
    per (method, rep) pair.  oracle_combined_loss_from_eig() replaces
    slogdet+inv with two O(p) dot products.

    Speedup vs old evaluate():
      - Data generation:    6× per rep → 1× per rep  (−83 %)
      - eigh calls:         6× per rep → 0×           (−100 %; from cache)
      - oracle_combined_loss: eliminated (slogdet+inv) → O(p) arithmetic
      - lw_nonlinear_shrinkage: O(p) Python loop → numpy broadcast
    """
    # Precompute true eigenvalues in descending order (once, O(p log p))
    true_eigs_desc = np.sort(np.linalg.eigvalsh(Sigma_0))[::-1]
    true_eigs_desc = np.maximum(true_eigs_desc, 1e-14)

    combined_lists: dict = {m: [] for m in METHODS}
    eigenval_lists: dict = {m: [] for m in METHODS}

    for rep, (lams, _V, tau_hat, M_diag) in enumerate(eigen_cache):
        # Generate X once per rep (needed only to fit LedoitWolf for LW-L)
        rng = np.random.default_rng(p * 10000 + n * 100 + rep)
        X = generate_data(n, p, Sigma_0, rng)

        # LW-L shrinkage coefficient from sklearn (O(n·p) — done once per rep)
        lw_model = LedoitWolf(store_precision=False, assume_centered=True).fit(X)
        alpha_lwl = float(lw_model.shrinkage_)
        mu_lwl = float(np.mean(lams))  # trace(S)/p

        for method in METHODS:
            rho = tuned_rhos.get(method, 0.0)

            # ── Compute shrunk eigenvalues (ascending) ──────────────────────
            if method == 'sample':
                s = lams  # already ascending, no copy needed
            elif method == 'LW-L':
                # Sigma_LWL = (1-alpha)*S + alpha*mu*I  shares eigenvectors with S
                s = (1.0 - alpha_lwl) * lams + alpha_lwl * mu_lwl
                s = np.maximum(s, 1e-12)
            elif method == 'LW-NL':
                # Vectorised broadcast replaces O(p) Python loop
                s = lw_nonlinear_shrinkage_eig(lams, n)
            elif method == 'W-SCOPE':
                s = _scope_w_eigenvalues(lams, tau_hat, rho)
            elif method == 'W-DRCOE':
                s = _wdrcoe_eigenvalues(lams, rho)
            elif method == 'WISE':
                s = _wise_eigenvalues(lams, rho)
            else:
                raise ValueError(f"Unknown method: {method}")

            # ── Oracle combined Stein-Frobenius loss (O(p), no inv/slogdet) ─
            if method == 'sample' and n <= p:
                cl = np.nan  # sample covariance is singular: loss undefined
            else:
                cl = oracle_combined_loss_from_eig(s, M_diag, tau_star)
                if not np.isfinite(cl):
                    cl = np.nan

            # ── Relative eigenvalue error (descending comparison) ───────────
            est_eigs_desc = s[::-1]  # eigh returns ascending; flip for comparison
            ev_err = float(np.mean(np.abs(est_eigs_desc - true_eigs_desc) / true_eigs_desc))

            combined_lists[method].append(cl)
            eigenval_lists[method].append(ev_err)

    # Average over reps (nanmean for combined loss to handle singular sample)
    combined_avgs: dict = {}
    eigenval_avgs: dict = {}
    for method in METHODS:
        arr = np.array(combined_lists[method], dtype=float)
        combined_avgs[method] = (
            float(np.nan) if np.all(np.isnan(arr)) else float(np.nanmean(arr))
        )
        eigenval_avgs[method] = float(np.mean(eigenval_lists[method]))

    return combined_avgs, eigenval_avgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    combined_records = {}   # col_key -> {method: value}
    eigenval_records = {}

    for p in [150, 200]:
        n_list = P_N_CONFIG[p]
        Sigma_0, _ = make_true_covariance(
            p=p, eigenvalues=np.arange(1, p + 1, dtype=float), seed=42
        )
        tau_star = float(p / np.sum(Sigma_0 ** 2))
        print(f"\n{'='*60}")
        print(f"p = {p}   (tau_star = {tau_star:.4g})")
        print(f"{'='*60}")

        for n in n_list:
            col_key = f"p={p},n={n}"
            print(f"\n  --- p={p}, n={n} ---")

            t0 = time.time()
            print(f"  [Tuning rho]")
            tuned_rhos, eigen_cache = tune_rho(p, n, Sigma_0, tau_star)
            t1 = time.time()
            print(f"  [Tuning done in {t1-t0:.1f}s]")

            print(f"  [Evaluating all methods]")
            combined_avgs, eigenval_avgs = evaluate(
                p, n, Sigma_0, tau_star, tuned_rhos, eigen_cache
            )
            t2 = time.time()
            print(f"  [Evaluation done in {t2-t1:.1f}s]")

            combined_records[col_key] = combined_avgs
            eigenval_records[col_key] = eigenval_avgs

    # Build DataFrames: rows = methods, columns = (p, n) configurations
    combined_df = pd.DataFrame(combined_records, index=METHODS)
    eigenval_df = pd.DataFrame(eigenval_records, index=METHODS)

    combined_df.index.name = 'Method'
    eigenval_df.index.name = 'Method'

    combined_path = RESULTS_DIR / 'exp2_combined_loss_table.csv'
    eigenval_path = RESULTS_DIR / 'exp2_eigenvalue_error_table.csv'
    combined_df.to_csv(combined_path)
    eigenval_df.to_csv(eigenval_path)
    print(f"\nSaved: {combined_path}")
    print(f"Saved: {eigenval_path}")

    # Print tables
    print("\n" + "="*70)
    print("Table: Combined Stein-Frobenius Loss")
    print("="*70)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(combined_df.to_string())

    print("\n" + "="*70)
    print("Table: Relative Eigenvalue Error")
    print("="*70)
    print(eigenval_df.to_string())


if __name__ == '__main__':
    main()
