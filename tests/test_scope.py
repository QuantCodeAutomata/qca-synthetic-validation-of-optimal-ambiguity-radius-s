"""
Comprehensive unit tests for src.scope.

Tests cover divergence functions, phi maps, the SCOPE estimator,
oracle loss functions, and the W-DRCOE / WISE baselines.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.scope import (
    kl_div,
    wasserstein_div,
    stein_sym_div,
    kl_phi,
    wasserstein_phi,
    stein_phi,
    scope_estimate,
    compute_rho_max,
    oracle_combined_loss,
    relative_eigenvalue_error,
    make_true_covariance,
    wdrcoe_estimate,
    wise_estimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_sigma(p: int = 5, seed: int = 0) -> np.ndarray:
    """Return a reproducible PD covariance matrix of size (p, p)."""
    Sigma_0, _ = make_true_covariance(p, eigenvalues=np.arange(1.0, p + 1.0), seed=seed)
    return Sigma_0


def _is_positive_definite(M: np.ndarray) -> bool:
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals > 0))


# ---------------------------------------------------------------------------
# 1. KL divergence properties
# ---------------------------------------------------------------------------

def test_kl_div_properties():
    test_pairs = [(1.0, 2.0), (3.0, 0.5), (0.1, 5.0), (2.0, 2.0), (7.0, 3.0)]
    for a, b in test_pairs:
        val = kl_div(a, b)
        assert val >= 0, f"kl_div({a}, {b}) = {val} < 0"

    # Self-divergence must be zero
    for a in [0.5, 1.0, 2.0, 10.0]:
        assert abs(kl_div(a, a)) < 1e-12, f"kl_div({a}, {a}) != 0"

    # Asymmetry: d(2, 1) > 0
    assert kl_div(2.0, 1.0) > 0


# ---------------------------------------------------------------------------
# 2. Wasserstein divergence properties
# ---------------------------------------------------------------------------

def test_wasserstein_div_properties():
    test_pairs = [(1.0, 4.0), (0.5, 2.0), (3.0, 3.0), (0.1, 0.9)]
    for a, b in test_pairs:
        val = wasserstein_div(a, b)
        assert val >= 0, f"wasserstein_div({a}, {b}) = {val} < 0"

    # Self-divergence
    for a in [0.5, 1.0, 3.0]:
        assert abs(wasserstein_div(a, a)) < 1e-12

    # Triangle-like: d(1, 4) <= d(1, 2) + d(2, 4) is generally NOT true,
    # but symmetry holds: d(a, b) == d(b, a)
    for a, b in [(1.0, 4.0), (2.0, 0.5)]:
        assert abs(wasserstein_div(a, b) - wasserstein_div(b, a)) < 1e-12, \
            "wasserstein_div is not symmetric"


# ---------------------------------------------------------------------------
# 3. Symmetrised Stein divergence properties
# ---------------------------------------------------------------------------

def test_stein_div_properties():
    test_pairs = [(1.0, 2.0), (4.0, 1.0), (0.5, 0.5), (3.0, 0.1)]
    for a, b in test_pairs:
        val = stein_sym_div(a, b)
        assert val >= 0, f"stein_sym_div({a}, {b}) = {val} < 0"

    # Self-divergence
    for a in [0.5, 1.0, 5.0]:
        assert abs(stein_sym_div(a, a)) < 1e-12

    # Symmetry: d_SS(a, b) == d_SS(b, a)
    for a, b in [(2.0, 3.0), (0.1, 4.0)]:
        assert abs(stein_sym_div(a, b) - stein_sym_div(b, a)) < 1e-12


# ---------------------------------------------------------------------------
# 4. KL phi at gamma=0 returns scalar target 1/sqrt(tau)
# ---------------------------------------------------------------------------

def test_kl_phi_scalar_target():
    """At gamma=0, kl_phi(tau, 0, b) == 1/sqrt(tau) for any b > 0."""
    for tau in [0.25, 1.0, 4.0, 9.0]:
        scalar_target = 1.0 / np.sqrt(tau)
        for b in [0.5, 1.0, 2.0, 5.0]:
            phi_val = kl_phi(tau, 0.0, b)
            assert abs(phi_val - scalar_target) < 1e-10, (
                f"kl_phi({tau}, 0, {b}) = {phi_val}, expected {scalar_target}"
            )


# ---------------------------------------------------------------------------
# 5. Wasserstein phi at small gamma ≈ 1/sqrt(tau)
# ---------------------------------------------------------------------------

def test_wasserstein_phi_scalar_target():
    """At gamma≈0 (1e-10), wasserstein_phi(tau, gamma, b) ≈ 1/sqrt(tau)."""
    gamma = 1e-10
    for tau in [0.25, 1.0, 4.0]:
        scalar_target = 1.0 / np.sqrt(tau)
        for b in [0.5, 1.0, 3.0]:
            phi_val = wasserstein_phi(tau, gamma, b)
            rel_err = abs(phi_val - scalar_target) / scalar_target
            assert rel_err < 1e-5, (
                f"wasserstein_phi({tau}, {gamma}, {b}) = {phi_val:.6g}, "
                f"expected ≈ {scalar_target:.6g} (rel err {rel_err:.2e})"
            )


# ---------------------------------------------------------------------------
# 6. scope_estimate returns positive-definite matrix
# ---------------------------------------------------------------------------

def test_scope_positive_definite():
    """scope_estimate is PD for various rho values and all three divergences."""
    Sigma_hat = _make_sample_sigma(p=5, seed=1)
    p = Sigma_hat.shape[0]
    tau = p / float(np.sum(Sigma_hat ** 2))

    for div in ('KL', 'W', 'SS'):
        for rho in (0.01, 0.1, 1.0, 10.0, 100.0):
            Sigma_star = scope_estimate(Sigma_hat, rho, divergence=div, tau=tau)
            assert _is_positive_definite(Sigma_star), (
                f"scope_estimate not PD for div={div}, rho={rho}"
            )


# ---------------------------------------------------------------------------
# 7. scope_estimate preserves eigenvectors of Sigma_hat
# ---------------------------------------------------------------------------

def test_scope_eigenvector_preservation():
    """The eigenvectors of scope_estimate match those of Sigma_hat (up to sign).

    Uses rho strictly below rho_max so the optimum is in the interior of the
    feasible set and eigenvalues remain distinct (eigenvectors well-defined).
    """
    from src.scope import compute_rho_max

    Sigma_hat = _make_sample_sigma(p=5, seed=2)
    p = Sigma_hat.shape[0]
    tau = p / float(np.sum(Sigma_hat ** 2))
    lambdas = np.linalg.eigvalsh(Sigma_hat)

    _, V_hat = np.linalg.eigh(Sigma_hat)

    for div in ('KL', 'W', 'SS'):
        rho_max = compute_rho_max(tau, lambdas, div)
        # Use rho well below rho_max so eigenvalues stay distinct
        rho = rho_max * 0.1
        Sigma_star = scope_estimate(Sigma_hat, rho=rho, divergence=div, tau=tau)
        _, V_star = np.linalg.eigh(Sigma_star)

        # |V_hat^T V_star| should be close to I (eigenvectors match up to sign)
        overlap = np.abs(V_hat.T @ V_star)
        off_diag = overlap - np.eye(p)
        assert np.max(np.abs(off_diag)) < 1e-6, (
            f"Eigenvectors not preserved for div={div}, rho={rho:.3g}; "
            f"max |off-diag| = {np.max(np.abs(off_diag)):.2e}"
        )


# ---------------------------------------------------------------------------
# 8. scope_estimate returns scalar-target matrix when rho >= rho_max
# ---------------------------------------------------------------------------

def test_scope_scalar_target_when_rho_geq_rho_max():
    """For rho >= rho_max, scope_estimate returns (1/sqrt(tau)) * I."""
    Sigma_hat = _make_sample_sigma(p=5, seed=3)
    p = Sigma_hat.shape[0]
    tau = p / float(np.sum(Sigma_hat ** 2))
    scalar_target = 1.0 / np.sqrt(tau)

    lambdas = np.linalg.eigvalsh(Sigma_hat)
    lambdas = np.maximum(lambdas, 1e-12)

    for div in ('KL', 'W', 'SS'):
        rho_max = compute_rho_max(tau, lambdas, div)
        # Use a rho clearly above rho_max
        rho_big = rho_max * 10.0 + 1.0
        Sigma_star = scope_estimate(Sigma_hat, rho_big, divergence=div, tau=tau)

        # Sigma_star should be proportional to I
        ratio = Sigma_star / scalar_target
        frob_err = float(np.linalg.norm(ratio - np.eye(p), 'fro'))
        assert frob_err < 1e-6, (
            f"div={div}: Sigma_star / scalar_target is not I "
            f"(||·-I||_F = {frob_err:.2e})"
        )


# ---------------------------------------------------------------------------
# 9. scope_estimate ≈ Sigma_hat as rho → 0
# ---------------------------------------------------------------------------

def test_scope_approaches_sample_cov_as_rho_to_zero():
    """scope_estimate(Sigma_hat, rho=1e-8) is close to Sigma_hat."""
    Sigma_hat = _make_sample_sigma(p=5, seed=4)
    p = Sigma_hat.shape[0]
    tau = p / float(np.sum(Sigma_hat ** 2))

    for div in ('KL', 'W', 'SS'):
        Sigma_star = scope_estimate(Sigma_hat, rho=1e-8, divergence=div, tau=tau)
        rel_err = (np.linalg.norm(Sigma_star - Sigma_hat, 'fro') /
                   np.linalg.norm(Sigma_hat, 'fro'))
        assert rel_err < 0.05, (
            f"div={div}: scope at rho=1e-8 not close to Sigma_hat "
            f"(rel err = {rel_err:.3f})"
        )


# ---------------------------------------------------------------------------
# 10. oracle_combined_loss minimum at scalar target when Sigma_0 ∝ I
# ---------------------------------------------------------------------------

def test_oracle_combined_loss_minimum_at_scalar_target():
    """When Sigma_0 = c*I, the combined loss is minimised at Sigma_est = c*I."""
    p = 5
    c = 2.0
    Sigma_0 = c * np.eye(p)
    tau_star = p / float(np.sum(Sigma_0 ** 2))   # = 1/c^2
    scalar_target = 1.0 / np.sqrt(tau_star)       # = c

    Sigma_opt = scalar_target * np.eye(p)
    Sigma_far = 2.0 * scalar_target * np.eye(p)

    loss_opt = oracle_combined_loss(Sigma_opt, Sigma_0, tau_star=tau_star)
    loss_far = oracle_combined_loss(Sigma_far, Sigma_0, tau_star=tau_star)

    assert np.isfinite(loss_opt), "Loss at optimal point is not finite"
    assert loss_opt <= loss_far, (
        f"Loss at scalar target ({loss_opt:.6g}) > loss at 2×target ({loss_far:.6g})"
    )


# ---------------------------------------------------------------------------
# 11. relative_eigenvalue_error = 0 for exact estimate
# ---------------------------------------------------------------------------

def test_relative_eigenvalue_error_zero_for_exact():
    """relative_eigenvalue_error(Sigma_0, Sigma_0) == 0."""
    Sigma_0 = _make_sample_sigma(p=6, seed=5)
    err = relative_eigenvalue_error(Sigma_0, Sigma_0)
    assert err == 0.0, f"Expected 0, got {err}"


# ---------------------------------------------------------------------------
# 12. make_true_covariance: orthogonal V and correct eigenvalues
# ---------------------------------------------------------------------------

def test_make_true_covariance_orthogonal():
    """V^T V = I and eigenvalues of Sigma_0 match the input eigenvalues."""
    p = 7
    eigenvalues = np.arange(1.0, p + 1.0)
    Sigma_0, V = make_true_covariance(p, eigenvalues=eigenvalues, seed=99)

    # V has orthonormal columns: V^T V = I
    VtV = V.T @ V
    frob = float(np.linalg.norm(VtV - np.eye(p), 'fro'))
    assert frob < 1e-10, f"V is not orthonormal (||V^T V - I||_F = {frob:.2e})"

    # Eigenvalues of Sigma_0 match the input eigenvalues (ascending order)
    eigs_out = np.sort(np.linalg.eigvalsh(Sigma_0))
    assert np.allclose(eigs_out, np.sort(eigenvalues), atol=1e-10), (
        f"Eigenvalues mismatch: got {eigs_out}, expected {eigenvalues}"
    )


# ---------------------------------------------------------------------------
# 13. All three divergences produce PD estimates
# ---------------------------------------------------------------------------

def test_all_divergences_produce_pd():
    """scope_estimate is PD for all divergences and a range of rho values."""
    Sigma_hat = _make_sample_sigma(p=8, seed=6)
    p = Sigma_hat.shape[0]
    tau = p / float(np.sum(Sigma_hat ** 2))

    for div in ('KL', 'W', 'SS'):
        for rho in np.logspace(-2, 2, 8):
            S = scope_estimate(Sigma_hat, float(rho), divergence=div, tau=tau)
            min_eig = float(np.min(np.linalg.eigvalsh(S)))
            assert min_eig > 0, (
                f"div={div}, rho={rho:.3g}: min eigenvalue = {min_eig:.3e}"
            )


# ---------------------------------------------------------------------------
# 14. wdrcoe_estimate returns PD matrix
# ---------------------------------------------------------------------------

def test_wdrcoe_estimate_pd():
    """wdrcoe_estimate returns a positive-definite covariance matrix."""
    Sigma_hat = _make_sample_sigma(p=5, seed=7)
    for rho in (0.1, 1.0, 10.0, 100.0):
        Sigma_w = wdrcoe_estimate(Sigma_hat, rho)
        assert _is_positive_definite(Sigma_w), (
            f"wdrcoe_estimate not PD for rho={rho}"
        )


# ---------------------------------------------------------------------------
# 15. wise_estimate returns PD matrix
# ---------------------------------------------------------------------------

def test_wise_estimate_pd():
    """wise_estimate returns a positive-definite covariance matrix."""
    Sigma_hat = _make_sample_sigma(p=5, seed=8)
    for rho in (0.1, 1.0, 5.0, 50.0):
        Sigma_w = wise_estimate(Sigma_hat, rho)
        assert _is_positive_definite(Sigma_w), (
            f"wise_estimate not PD for rho={rho}"
        )
