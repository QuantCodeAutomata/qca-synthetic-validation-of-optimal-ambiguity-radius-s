"""
Smoke tests for experiments 1, 2, and 4.

Each test runs a mini inline version of the corresponding experiment using
synthetic data only (no network calls, no external files).  The goal is to
verify that the core computational pipeline runs without errors and produces
numerically sensible results.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.scope import (
    scope_estimate,
    oracle_combined_loss,
    make_true_covariance,
    compute_rho_max,
    relative_eigenvalue_error,
    wdrcoe_estimate,
    wise_estimate,
)
from src.benchmark_estimators import (
    lw_linear_shrinkage,
    lw_nonlinear_shrinkage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_cov_zero_mean(X: np.ndarray) -> np.ndarray:
    """Zero-mean second-moment estimator: Sigma_hat = X^T X / n."""
    S = X.T @ X / X.shape[0]
    return (S + S.T) / 2.0


def _scope_tau(Sigma_hat: np.ndarray) -> float:
    p = Sigma_hat.shape[0]
    return p / float(np.sum(Sigma_hat ** 2))


def _generate_data(Sigma_0: np.ndarray, n: int, rng: np.random.Generator):
    """Draw n zero-mean observations from N(0, Sigma_0)."""
    p = Sigma_0.shape[0]
    L = np.linalg.cholesky(Sigma_0)
    Z = rng.standard_normal((n, p))
    return Z @ L.T


# ---------------------------------------------------------------------------
# Exp 1 — Radius scaling
# ---------------------------------------------------------------------------

def test_exp1_radius_scaling_runs():
    """Mini radius-scaling experiment: p=5, n=[10,20], 5 reps, 10 rho grid."""
    p = 5
    n_list = [10, 20]
    n_reps = 5
    rho_grid = np.logspace(-2, 2, 10)
    eigenvalues = np.arange(1.0, p + 1.0)
    Sigma_0, _ = make_true_covariance(p, eigenvalues=eigenvalues, seed=42)
    rng = np.random.default_rng(0)

    # results[n][rho_idx] = mean oracle loss over reps
    results: dict = {}
    for n in n_list:
        losses_by_rho = []
        for rho in rho_grid:
            rep_losses = []
            for _ in range(n_reps):
                X = _generate_data(Sigma_0, n, rng)
                Sigma_hat = _sample_cov_zero_mean(X)
                tau = _scope_tau(Sigma_hat)
                Sigma_est = scope_estimate(
                    Sigma_hat, rho=float(rho), divergence='W', tau=tau
                )
                loss = oracle_combined_loss(Sigma_est, Sigma_0)
                rep_losses.append(loss)
            losses_by_rho.append(float(np.mean(rep_losses)))
        results[n] = losses_by_rho

    # All losses must be finite
    for n, losses in results.items():
        assert all(np.isfinite(l) for l in losses), (
            f"Non-finite loss for n={n}: {losses}"
        )

    # There must be a finite minimiser in the grid (not all inf)
    for n, losses in results.items():
        assert min(losses) < np.inf, f"All losses infinite for n={n}"

    # Radius scaling: as n grows, the minimum loss should decrease
    min_losses = {n: min(results[n]) for n in n_list}
    assert min_losses[20] <= min_losses[10] * 1.5, (
        "Expected loss to decrease (or not increase much) as n grows; "
        f"got n=10: {min_losses[10]:.4f}, n=20: {min_losses[20]:.4f}"
    )


# ---------------------------------------------------------------------------
# Exp 2 — High-dimensional comparison
# ---------------------------------------------------------------------------

def test_exp2_hd_comparison_runs():
    """Mini HD comparison: p=10, n=[8,15], 3 reps.

    When n < p (n=8, p=10) the sample covariance is rank-deficient; only the
    regularised estimators (LW-L, LW-NL, SCOPE variants, W-DRCOE, WISE) are
    required to be positive-definite.  For n >= p all estimators must be PD.
    """
    p = 10
    n_list = [8, 15]
    n_reps = 3
    eigenvalues = np.arange(1.0, p + 1.0)
    Sigma_0, _ = make_true_covariance(p, eigenvalues=eigenvalues, seed=0)
    rng = np.random.default_rng(1)

    regularised = ['LW-L', 'LW-NL', 'W-SCOPE', 'KL-SCOPE', 'SS-SCOPE',
                   'W-DRCOE', 'WISE']
    methods = ['sample'] + regularised

    rho_default = 1.0
    reg_losses: dict = {m: [] for m in regularised}

    for n in n_list:
        for _ in range(n_reps):
            X = _generate_data(Sigma_0, n, rng)
            Sigma_hat = _sample_cov_zero_mean(X)
            tau = _scope_tau(Sigma_hat)

            estimates = {
                'sample': Sigma_hat.copy(),
                'LW-L': lw_linear_shrinkage(X, assume_centered=True),
                'LW-NL': lw_nonlinear_shrinkage(X, assume_centered=True),
                'W-SCOPE': scope_estimate(
                    Sigma_hat, rho_default, divergence='W', tau=tau),
                'KL-SCOPE': scope_estimate(
                    Sigma_hat, rho_default, divergence='KL', tau=tau),
                'SS-SCOPE': scope_estimate(
                    Sigma_hat, rho_default, divergence='SS', tau=tau),
                'W-DRCOE': wdrcoe_estimate(Sigma_hat, rho_default),
                'WISE': wise_estimate(Sigma_hat, rho_default),
            }

            for method, est in estimates.items():
                min_eig = float(np.min(np.linalg.eigvalsh(est)))
                if method == 'sample' and n < p:
                    # Rank-deficient in the HD regime — allow numerical near-zero
                    assert min_eig > -1e-10, (
                        f"sample cov has large negative eigenvalue "
                        f"({min_eig:.3e}), n={n}"
                    )
                else:
                    assert min_eig > 0, (
                        f"{method} not PD (min eig={min_eig:.3e}), n={n}"
                    )

            # Oracle loss: only compute for regularised estimators (always PD)
            for method in regularised:
                loss = oracle_combined_loss(estimates[method], Sigma_0)
                assert np.isfinite(loss), (
                    f"oracle_combined_loss not finite for {method}, n={n}"
                )
                reg_losses[method].append(loss)

    expected = n_reps * len(n_list)
    for method, losses in reg_losses.items():
        assert len(losses) == expected, (
            f"{method}: expected {expected} loss values, got {len(losses)}"
        )
        assert all(np.isfinite(l) for l in losses), (
            f"{method}: some losses not finite"
        )


# ---------------------------------------------------------------------------
# Exp 4 — A/B testing
# ---------------------------------------------------------------------------

def test_exp4_ab_testing_runs():
    """Mini A/B test: p=5, n=[6,8], 3 reps, 5 training pairs, 10 test pairs."""
    p = 5
    n_list = [6, 8]
    n_reps = 3
    n_train_pairs = 5
    n_test_pairs = 10
    rho_grid = np.logspace(-1, 1, 5)

    Sigma_0, _ = make_true_covariance(p, eigenvalues=np.arange(1.0, p + 1.0), seed=7)
    rng = np.random.default_rng(42)

    div_map = {'W-SCOPE': 'W', 'KL-SCOPE': 'KL', 'SS-SCOPE': 'SS'}
    scope_methods = list(div_map.keys())

    def run_once(n: int) -> dict:
        # --- Training: pick best rho per SCOPE method by oracle loss ---
        best_rho: dict = {}
        for method in scope_methods:
            div = div_map[method]
            rho_losses = []
            for rho in rho_grid:
                pair_losses = []
                for _ in range(n_train_pairs):
                    X = _generate_data(Sigma_0, n, rng)
                    Sigma_hat = _sample_cov_zero_mean(X)
                    tau = _scope_tau(Sigma_hat)
                    est = scope_estimate(
                        Sigma_hat, float(rho), divergence=div, tau=tau
                    )
                    pair_losses.append(oracle_combined_loss(est, Sigma_0))
                rho_losses.append(float(np.mean(pair_losses)))
            best_rho[method] = float(rho_grid[int(np.argmin(rho_losses))])

        # --- Test: compare tuned SCOPE vs sample on held-out pairs ---
        test_losses: dict = {m: [] for m in scope_methods + ['sample']}
        for _ in range(n_test_pairs):
            X = _generate_data(Sigma_0, n, rng)
            Sigma_hat = _sample_cov_zero_mean(X)
            tau = _scope_tau(Sigma_hat)

            test_losses['sample'].append(oracle_combined_loss(Sigma_hat, Sigma_0))

            for method in scope_methods:
                div = div_map[method]
                est = scope_estimate(
                    Sigma_hat, best_rho[method], divergence=div, tau=tau
                )
                test_losses[method].append(oracle_combined_loss(est, Sigma_0))

        return {m: float(np.mean(v)) for m, v in test_losses.items()}

    for n in n_list:
        for _ in range(n_reps):
            avg_losses = run_once(n)

            # All average losses must be finite and positive
            for method, avg_loss in avg_losses.items():
                assert np.isfinite(avg_loss), (
                    f"n={n}, {method}: avg test loss is not finite ({avg_loss})"
                )
                assert avg_loss > 0, (
                    f"n={n}, {method}: avg test loss is non-positive ({avg_loss})"
                )

            # Tuned SCOPE should not be catastrophically worse than sample
            sample_loss = avg_losses['sample']
            for method in scope_methods:
                ratio = avg_losses[method] / sample_loss
                assert ratio < 10.0, (
                    f"n={n}, {method}: SCOPE loss ({avg_losses[method]:.4f}) is "
                    f">10× the sample loss ({sample_loss:.4f})"
                )
