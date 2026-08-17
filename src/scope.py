"""
SCOPE (Spectral Covariance with Optimal Perturbation Estimator) module.

Implements the joint covariance-precision SCOPE estimator family for three divergences:
  - KL-SCOPE   : Kullback-Leibler divergence
  - W-SCOPE    : Wasserstein (squared Bures) divergence
  - SS-SCOPE   : Symmetrized Stein divergence

Reference: Paper on optimal ambiguity radius scaling for SCOPE estimators.

Key computational simplification: eigenvalue shrinkage via convex spectral
reformulation - eigenvectors are preserved from Sigma_hat; only eigenvalues
are shrunk toward the scalar target sqrt(1/tau).

Custom implementation — Context7 found no library equivalent for SCOPE
estimators with the combined Stein-Frobenius loss and spectral reformulation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Divergence functions  d(a, b)
# ---------------------------------------------------------------------------

def kl_div(a: float, b: float) -> float:
    """KL divergence  d_KL(a, b) = 0.5*(a/b - 1 - log(a/b)).

    Parameters
    ----------
    a, b : float  (positive scalars)
    """
    return 0.5 * (a / b - 1.0 - np.log(a / b))


def wasserstein_div(a: float, b: float) -> float:
    """Wasserstein (squared-Bures) divergence  d_W(a, b) = a + b - 2*sqrt(a*b).

    Parameters
    ----------
    a, b : float  (positive scalars)
    """
    return a + b - 2.0 * np.sqrt(a * b)


def stein_sym_div(a: float, b: float) -> float:
    """Symmetrised Stein divergence  d_SS(a, b) = 0.5*(b/a + a/b - 2).

    Parameters
    ----------
    a, b : float  (positive scalars)
    """
    return 0.5 * (b / a + a / b - 2.0)


# Vectorised wrappers
def kl_div_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b  # 0/b → 0 (→ inf KL), a/0 → inf (→ inf KL)
        val = 0.5 * (r - 1.0 - np.log(r))
    # limit: r → 0  ⇒ KL → +∞;  r → ∞ ⇒ KL → +∞
    return np.where(np.isfinite(val), val, np.inf)


def wasserstein_div_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b - 2.0 * np.sqrt(a * b)


def stein_sym_div_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (b / a + a / b - 2.0)


# ---------------------------------------------------------------------------
# phi(tau, gamma, b) maps : shrinkage target per eigenvalue
# ---------------------------------------------------------------------------

def kl_phi(tau: float, gamma: float, b: float) -> float:
    """Closed-form phi for KL-SCOPE.

    phi = (-gamma + sqrt(gamma^2 + 8*tau*(gamma+2)*b^2)) / (4*tau*b)

    Derived from KKT condition of:
      min -log s + b/s + tau*(s - b)^2  +  gamma * d_KL(s, b)
    Paper provides this as the closed-form eigenvalue map.
    """
    num = -gamma + np.sqrt(gamma ** 2 + 8.0 * tau * (gamma + 2.0) * b ** 2)
    return num / (4.0 * tau * b)


def wasserstein_phi(tau: float, gamma: float, b: float) -> float:
    """Numerical phi for W-SCOPE.

    Unique positive root of: tau*a^2 + gamma*a - gamma*sqrt(b)*sqrt(a) - 1 = 0

    Bracket: as gamma→∞, a→b; as gamma→0, a→1/sqrt(tau).
    """
    def equation(a: float) -> float:
        return tau * a ** 2 + gamma * a - gamma * np.sqrt(b) * np.sqrt(a) - 1.0

    # Bracket robustly
    lo = 1e-12
    target_lo = 1.0 / np.sqrt(tau)
    target_hi = b
    hi = max(target_lo, target_hi) * 10.0 + 10.0

    # Ensure sign change
    if equation(lo) * equation(hi) > 0:
        hi = hi * 100.0 + 1e6
    if equation(lo) * equation(hi) > 0:
        return b  # fallback to sample eigenvalue

    return brentq(equation, lo, hi, xtol=1e-12, rtol=1e-10)


def stein_phi(tau: float, gamma: float, b: float) -> float:
    """Numerical phi for SS-SCOPE.

    Unique positive root of: 2*tau*b*a^3 + gamma*a^2 - 2*b*a - gamma*b^2 = 0

    Bracket: as gamma→∞, a→b; as gamma→0, a→1/sqrt(tau).
    """
    def equation(a: float) -> float:
        return 2.0 * tau * b * a ** 3 + gamma * a ** 2 - 2.0 * b * a - gamma * b ** 2

    lo = 1e-12
    hi = max(1.0 / np.sqrt(tau), b) * 10.0 + 10.0

    if equation(lo) * equation(hi) > 0:
        hi = hi * 100.0 + 1e6
    if equation(lo) * equation(hi) > 0:
        return b

    return brentq(equation, lo, hi, xtol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Vectorised bisection helper
# ---------------------------------------------------------------------------

def _vbisect(
    f: "Callable[[np.ndarray], np.ndarray]",
    lo: np.ndarray,
    hi: np.ndarray,
    n_iter: int = 52,
) -> np.ndarray:
    """Vectorised bisection over an array of brackets.

    Solves f(a) = 0 for each element simultaneously, where f is vectorised
    (accepts and returns ndarray of the same shape as lo/hi).

    Assumptions: f(lo) <= 0 and f(hi) >= 0 element-wise (sign-change bracket).
    52 iterations yield relative precision < 2^{-52} ≈ 2.2e-16 (double precision).
    """
    lo, hi = lo.copy(), hi.copy()
    f_lo = f(lo)
    for _ in range(n_iter):
        mid = (lo + hi) * 0.5
        f_mid = f(mid)
        # Root lies in [lo, mid] where f_lo and f_mid have opposite signs
        left = f_lo * f_mid <= 0.0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# Vector versions of phi
# ---------------------------------------------------------------------------

def kl_phi_v(tau: float, gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """Vectorised KL phi (closed-form)."""
    num = -gamma + np.sqrt(gamma ** 2 + 8.0 * tau * (gamma + 2.0) * lambdas ** 2)
    return num / (4.0 * tau * lambdas)


def wasserstein_phi_v(tau: float, gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """Vectorised W phi via guarded Newton's method across all eigenvalues.

    Equation per eigenvalue b:  f(a) = tau*a^2 + gamma*a - gamma*sqrt(b)*sqrt(a) - 1 = 0
    f'(a) = 2*tau*a + gamma - 0.5*gamma*sqrt(b)/sqrt(a)

    Newton from a0 = midpoint of scalar target and eigenvalue converges
    quadratically, typically in 8–12 iterations vs the 52 needed for bisection.
    Steps that produce non-positive values are halved (backtracking).
    """
    if gamma == 0.0:
        return np.full_like(lambdas, 1.0 / np.sqrt(tau))

    sq_b = np.sqrt(lambdas)
    a_scalar = 1.0 / np.sqrt(tau)
    # Initialise between scalar target and eigenvalue (phi lies in this interval)
    a = np.where(lambdas > a_scalar,
                 0.5 * (a_scalar + lambdas),
                 np.where(lambdas < a_scalar,
                          0.5 * (lambdas + a_scalar),
                          a_scalar * np.ones_like(lambdas)))
    a = np.maximum(a, 1e-14)

    for _ in range(8):
        sq_a = np.sqrt(a)
        f  = tau * a * a + gamma * a - gamma * sq_b * sq_a - 1.0
        df = 2.0 * tau * a + gamma - 0.5 * gamma * sq_b / sq_a
        # Safeguard: if df ≈ 0, skip step
        step = np.where(np.abs(df) > 1e-30, f / df, 0.0)
        a_new = a - step
        # Backtrack if step overshoots into non-positive territory
        a_new = np.where(a_new <= 0.0, a * 0.5, a_new)
        a = a_new

    return np.maximum(a, 1e-12)


def stein_phi_v(tau: float, gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """Vectorised SS phi via guarded Newton's method across all eigenvalues.

    Equation per eigenvalue b:
      f(a) = 2*tau*b*a^3 + gamma*a^2 - 2*b*a - gamma*b^2 = 0
    f'(a) = 6*tau*b*a^2 + 2*gamma*a - 2*b

    Newton converges in 8–12 iterations; same safeguarding as wasserstein_phi_v.
    """
    if gamma == 0.0:
        return np.full_like(lambdas, 1.0 / np.sqrt(tau))

    b = lambdas
    a_scalar = 1.0 / np.sqrt(tau)
    a = np.where(b > a_scalar,
                 0.5 * (a_scalar + b),
                 np.where(b < a_scalar,
                          0.5 * (b + a_scalar),
                          a_scalar * np.ones_like(b)))
    a = np.maximum(a, 1e-14)

    for _ in range(8):
        f  = 2.0 * tau * b * a ** 3 + gamma * a ** 2 - 2.0 * b * a - gamma * b ** 2
        df = 6.0 * tau * b * a ** 2 + 2.0 * gamma * a - 2.0 * b
        step = np.where(np.abs(df) > 1e-30, f / df, 0.0)
        a_new = a - step
        a_new = np.where(a_new <= 0.0, a * 0.5, a_new)
        a = a_new

    return np.maximum(a, 1e-12)


# ---------------------------------------------------------------------------
# Batch phi functions  — process K gamma values × p eigenvalues in one call
# ---------------------------------------------------------------------------

def kl_phi_batch(tau: float, gammas: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Closed-form KL phi over a (K,) gamma vector and (p,) lambda vector.

    Returns an (K, p) array where result[k, j] = phi(tau, gammas[k], lambdas[j]).
    Broadcasting eliminates the Python loop over gammas.
    """
    g = gammas[:, None]   # (K, 1)
    b = lambdas[None, :]  # (1, p)
    num = -g + np.sqrt(g ** 2 + 8.0 * tau * (g + 2.0) * b ** 2)
    return num / (4.0 * tau * b)   # (K, p)


def wasserstein_phi_batch(
    tau: float, gammas: np.ndarray, lambdas: np.ndarray, n_iter: int = 12
) -> np.ndarray:
    """Vectorised W phi via Newton's method for all (gamma, lambda) pairs.

    Returns an (K, p) array where result[k, j] = phi(tau, gammas[k], lambdas[j]).
    Newton iterations are broadcast simultaneously over the full (K, p) grid.

    Equation: f(a) = tau*a^2 + gamma*a - gamma*sqrt(b)*sqrt(a) - 1 = 0
    f'(a)    = 2*tau*a + gamma - 0.5*gamma*sqrt(b)/sqrt(a)
    """
    g = gammas[:, None]        # (K, 1)
    b = lambdas[None, :]       # (1, p)
    sq_b = np.sqrt(b)          # (1, p)
    a_scalar = 1.0 / np.sqrt(tau)

    # Initialise at midpoint between scalar target and each eigenvalue
    a = np.broadcast_to(0.5 * (a_scalar + lambdas[None, :]),
                        (len(gammas), len(lambdas))).copy()
    a = np.maximum(a, 1e-14)

    for _ in range(n_iter):
        sq_a = np.sqrt(a)                                              # (K, p)
        f  = tau * a * a + g * a - g * sq_b * sq_a - 1.0             # (K, p)
        df = 2.0 * tau * a + g - 0.5 * g * sq_b / np.maximum(sq_a, 1e-30)
        step = np.where(np.abs(df) > 1e-30, f / df, 0.0)
        a_new = a - step
        a_new = np.where(a_new <= 0.0, a * 0.5, a_new)
        a = a_new

    return np.maximum(a, 1e-12)


def stein_phi_batch(
    tau: float, gammas: np.ndarray, lambdas: np.ndarray, n_iter: int = 12
) -> np.ndarray:
    """Vectorised SS phi via Newton's method for all (gamma, lambda) pairs.

    Returns an (K, p) array where result[k, j] = phi(tau, gammas[k], lambdas[j]).

    Equation: f(a) = 2*tau*b*a^3 + gamma*a^2 - 2*b*a - gamma*b^2 = 0
    f'(a)    = 6*tau*b*a^2 + 2*gamma*a - 2*b
    """
    g = gammas[:, None]   # (K, 1)
    b = lambdas[None, :]  # (1, p)
    a_scalar = 1.0 / np.sqrt(tau)

    a = np.broadcast_to(0.5 * (a_scalar + lambdas[None, :]),
                        (len(gammas), len(lambdas))).copy()
    a = np.maximum(a, 1e-14)

    for _ in range(n_iter):
        f  = 2.0 * tau * b * a ** 3 + g * a ** 2 - 2.0 * b * a - g * b ** 2
        df = 6.0 * tau * b * a ** 2 + 2.0 * g * a - 2.0 * b
        step = np.where(np.abs(df) > 1e-30, f / df, 0.0)
        a_new = a - step
        a_new = np.where(a_new <= 0.0, a * 0.5, a_new)
        a = a_new

    return np.maximum(a, 1e-12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_rho_max(
    tau: float,
    lambdas: np.ndarray,
    divergence: Literal["KL", "W", "SS"],
) -> float:
    """Compute rho_max = sum_i d(sqrt(1/tau), lambda_hat_i).

    When rho >= rho_max the unconstrained minimiser (scalar target) lies inside
    the feasible set and Sigma_star = sqrt(1/tau) * I.
    """
    scalar_target = 1.0 / np.sqrt(tau)
    if divergence == "KL":
        return float(np.sum(kl_div_v(scalar_target, lambdas)))
    elif divergence == "W":
        return float(np.sum(wasserstein_div_v(scalar_target, lambdas)))
    elif divergence == "SS":
        return float(np.sum(stein_sym_div_v(scalar_target, lambdas)))
    else:
        raise ValueError(f"Unknown divergence: {divergence}")


def _gamma_equation(
    gamma: float,
    tau: float,
    rho: float,
    lambdas: np.ndarray,
    divergence: Literal["KL", "W", "SS"],
) -> float:
    """Residual of the dual constraint: sum_i d(phi_i, lambda_i) - rho = 0."""
    if divergence == "KL":
        phi = kl_phi_v(tau, gamma, lambdas)
        return float(np.sum(kl_div_v(phi, lambdas))) - rho
    elif divergence == "W":
        phi = wasserstein_phi_v(tau, gamma, lambdas)
        return float(np.sum(wasserstein_div_v(phi, lambdas))) - rho
    elif divergence == "SS":
        phi = stein_phi_v(tau, gamma, lambdas)
        return float(np.sum(stein_sym_div_v(phi, lambdas))) - rho
    else:
        raise ValueError(f"Unknown divergence: {divergence}")


def solve_gamma_star(
    tau: float,
    rho: float,
    lambdas: np.ndarray,
    divergence: Literal["KL", "W", "SS"],
) -> float:
    """Solve for gamma* >= 0 s.t. sum_i d(phi(tau,gamma*,lambda_i), lambda_i) = rho.

    Uses brentq on [0, gamma_hi] where gamma_hi is chosen large enough so that
    the sum of divergences drops below rho.

    # Custom — Context7 found no library equivalent (paper dual equation)
    """
    # At gamma=0, phi = scalar target => sum d = rho_max >= rho (by assumption here)
    # At gamma→inf, phi → lambda_hat => sum d → 0 < rho
    gamma_hi = 1.0
    while _gamma_equation(gamma_hi, tau, rho, lambdas, divergence) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    if _gamma_equation(0.0, tau, rho, lambdas, divergence) < 0:
        return 0.0
    if _gamma_equation(gamma_hi, tau, rho, lambdas, divergence) > 0:
        return gamma_hi

    gamma_star = brentq(
        _gamma_equation,
        a=0.0,
        b=gamma_hi,
        args=(tau, rho, lambdas, divergence),
        xtol=1e-10,
        rtol=1e-10,
    )
    return float(gamma_star)


# ---------------------------------------------------------------------------
# Main SCOPE estimator
# ---------------------------------------------------------------------------

def scope_estimate(
    Sigma_hat: np.ndarray,
    rho: float,
    divergence: Literal["KL", "W", "SS"] = "W",
    tau: Optional[float] = None,
) -> np.ndarray:
    """Compute the SCOPE covariance estimate.

    Implements the joint covariance-precision estimator via eigenvalue shrinkage.
    Eigenvectors are preserved from Sigma_hat; eigenvalues are shrunk toward
    the scalar target sqrt(1/tau) according to the chosen divergence ball.

    Parameters
    ----------
    Sigma_hat : (p, p) ndarray
        Nominal/sample covariance matrix (symmetric PSD).
    rho : float
        Ambiguity radius (>= 0).
    divergence : {"KL", "W", "SS"}
        Divergence type: KL, Wasserstein, or Symmetrised Stein.
    tau : float or None
        Precision parameter.  If None, uses plug-in tau_hat = p / ||Sigma_hat||_F^2.

    Returns
    -------
    Sigma_star : (p, p) ndarray
        SCOPE estimate (symmetric, positive definite).
    """
    p = Sigma_hat.shape[0]
    assert Sigma_hat.shape == (p, p), "Sigma_hat must be square."

    # Eigendecomposition in ascending order
    lambdas, V = np.linalg.eigh(Sigma_hat)
    lambdas = np.maximum(lambdas, 1e-12)  # numerical floor only

    # Plug-in tau if not provided
    if tau is None:
        tau = p / np.sum(Sigma_hat ** 2)

    scalar_target = 1.0 / np.sqrt(tau)

    # Compute rho_max
    rho_max = compute_rho_max(tau, lambdas, divergence)

    if rho >= rho_max:
        # Trivial solution: scalar target
        Sigma_star = scalar_target * np.eye(p)
    else:
        # Solve for dual variable gamma*
        gamma_star = solve_gamma_star(tau, rho, lambdas, divergence)

        # Compute shrunk eigenvalues
        if divergence == "KL":
            s_star = kl_phi_v(tau, gamma_star, lambdas)
        elif divergence == "W":
            s_star = wasserstein_phi_v(tau, gamma_star, lambdas)
        elif divergence == "SS":
            s_star = stein_phi_v(tau, gamma_star, lambdas)
        else:
            raise ValueError(f"Unknown divergence: {divergence}")

        s_star = np.maximum(s_star, 1e-12)
        Sigma_star = V @ np.diag(s_star) @ V.T

    # Symmetric reconstruction to control numerical drift
    Sigma_star = (Sigma_star + Sigma_star.T) / 2.0
    return Sigma_star


def scope_estimate_from_eig(
    lambdas: np.ndarray,
    V: np.ndarray,
    tau: float,
    rho: float,
    divergence: Literal["KL", "W", "SS"] = "W",
    rho_max: Optional[float] = None,
) -> np.ndarray:
    """SCOPE estimate from a precomputed eigendecomposition.

    Avoids the cost of `np.linalg.eigh` when the same nominal covariance is
    evaluated at many rho values (e.g. inside `tune_rho`).

    Parameters
    ----------
    lambdas   : (p,) ascending eigenvalues (floored at 1e-12).
    V         : (p, p) eigenvector matrix from `np.linalg.eigh`.
    tau       : precision parameter p / ||Sigma_hat||_F^2.
    rho       : ambiguity radius.
    divergence: {"KL", "W", "SS"}.
    rho_max   : precomputed sum_i d(sqrt(1/tau), lambda_i); computed here if None.

    Returns
    -------
    Sigma_star : (p, p) symmetric SCOPE estimate.
    """
    p = len(lambdas)
    scalar_target = 1.0 / np.sqrt(tau)

    if rho_max is None:
        rho_max = compute_rho_max(tau, lambdas, divergence)

    if rho >= rho_max:
        Sigma_star = scalar_target * np.eye(p)
    else:
        gamma_star = solve_gamma_star(tau, rho, lambdas, divergence)

        if divergence == "KL":
            s_star = kl_phi_v(tau, gamma_star, lambdas)
        elif divergence == "W":
            s_star = wasserstein_phi_v(tau, gamma_star, lambdas)
        elif divergence == "SS":
            s_star = stein_phi_v(tau, gamma_star, lambdas)
        else:
            raise ValueError(f"Unknown divergence: {divergence}")

        s_star = np.maximum(s_star, 1e-12)
        Sigma_star = V @ np.diag(s_star) @ V.T

    return (Sigma_star + Sigma_star.T) / 2.0


# ---------------------------------------------------------------------------
# Oracle loss functions
# ---------------------------------------------------------------------------

def oracle_combined_loss(
    Sigma_est: np.ndarray,
    Sigma_0: np.ndarray,
    tau_star: Optional[float] = None,
) -> float:
    """Oracle combined Stein-Frobenius loss.

    ell(Sigma_est, Sigma_0) =
        -log det(Sigma_est^{-1}) + <Sigma_est^{-1}, Sigma_0>
        + (tau*/2) * (||Sigma_est||_F^2 - 2 * <Sigma_est, Sigma_0>)

    Note: -log det(Sigma_est^{-1}) = log det(Sigma_est).

    Parameters
    ----------
    Sigma_est : (p, p) ndarray  Estimated covariance.
    Sigma_0   : (p, p) ndarray  True covariance.
    tau_star  : float or None   Oracle tau = p / ||Sigma_0||_F^2. If None, computed here.

    Returns
    -------
    float : loss value (lower is better).
    """
    p = Sigma_est.shape[0]

    if tau_star is None:
        tau_star = p / np.sum(Sigma_0 ** 2)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(Sigma_est)
    if np.any(eigvals <= 0):
        return np.inf

    sign, logdet = np.linalg.slogdet(Sigma_est)
    if sign <= 0:
        return np.inf

    Sigma_inv = np.linalg.inv(Sigma_est)

    stein_loss = logdet + np.trace(Sigma_inv @ Sigma_0)
    frob_term = (tau_star / 2.0) * (
        np.sum(Sigma_est ** 2) - 2.0 * np.trace(Sigma_est @ Sigma_0)
    )

    return float(stein_loss + frob_term)


def oracle_frobenius_loss(Sigma_est: np.ndarray, Sigma_0: np.ndarray) -> float:
    """Oracle Frobenius loss ||Sigma_est - Sigma_0||_F^2."""
    diff = Sigma_est - Sigma_0
    return float(np.sum(diff ** 2))


def oracle_stein_loss(Sigma_est: np.ndarray, Sigma_0: np.ndarray) -> float:
    """Oracle Stein loss: log det Sigma_est + tr(Sigma_est^{-1} Sigma_0).

    Returns np.inf if Sigma_est is not positive definite.
    """
    eigvals = np.linalg.eigvalsh(Sigma_est)
    if np.any(eigvals <= 0):
        return np.inf
    sign, logdet = np.linalg.slogdet(Sigma_est)
    if sign <= 0:
        return np.inf
    Sigma_inv = np.linalg.inv(Sigma_est)
    return float(logdet + np.trace(Sigma_inv @ Sigma_0))


def relative_eigenvalue_error(
    Sigma_est: np.ndarray,
    Sigma_0: np.ndarray,
) -> float:
    """Relative eigenvalue estimation error.

    (1/p) * sum_i |lambda_est_i - lambda_0_i| / lambda_0_i
    where eigenvalues are sorted in decreasing order.
    """
    p = Sigma_est.shape[0]
    est_eigs = np.sort(np.linalg.eigvalsh(Sigma_est))[::-1]
    true_eigs = np.sort(np.linalg.eigvalsh(Sigma_0))[::-1]
    true_eigs = np.maximum(true_eigs, 1e-12)
    return float(np.mean(np.abs(est_eigs - true_eigs) / true_eigs))


# ---------------------------------------------------------------------------
# Reproducible true covariance generator
# ---------------------------------------------------------------------------

def make_true_covariance(
    p: int,
    eigenvalues: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible true covariance Sigma_0 = V^T Lambda V.

    Uses QR decomposition of a Gaussian matrix with sign-normalised diagonal
    of R to fix the orthogonal matrix V uniquely.

    Parameters
    ----------
    p : int               Dimension.
    eigenvalues : ndarray Diagonal of Lambda (ascending). Defaults to [1,...,p].
    seed : int            Random seed.

    Returns
    -------
    Sigma_0 : (p, p) ndarray  True covariance matrix.
    V       : (p, p) ndarray  Orthogonal matrix (rows are eigenvectors).
    """
    rng = np.random.default_rng(seed)

    if eigenvalues is None:
        eigenvalues = np.arange(1, p + 1, dtype=float)

    # Generate V via QR decomposition with sign normalisation
    G = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(G)
    # Sign normalisation: ensure diagonal of R is positive
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]  # V = Q (p columns are eigenvectors)

    Lambda = np.diag(eigenvalues)
    Sigma_0 = Q @ Lambda @ Q.T
    Sigma_0 = (Sigma_0 + Sigma_0.T) / 2.0

    return Sigma_0, Q


# ---------------------------------------------------------------------------
# W-DRCOE and WISE estimators (competing baselines)
# ---------------------------------------------------------------------------

def _wdrcoe_phi_equation(gamma: float, b: float) -> float:
    """KKT condition for W-DRCOE: gamma*u^2 - gamma*sqrt(b)*u - 1 = 0, u=sqrt(s)."""
    if gamma <= 0:
        return np.inf
    disc = gamma ** 2 * b + 4.0 * gamma
    if disc < 0:
        return 1e-12
    u = (gamma * np.sqrt(b) + np.sqrt(disc)) / (2.0 * gamma)
    return u ** 2  # = s_i


def wdrcoe_eigenvalue(gamma: float, b: float) -> float:
    """Scalar W-DRCOE eigenvalue (kept for backward compatibility)."""
    if gamma <= 0:
        return np.inf
    u = (np.sqrt(b) + np.sqrt(b + 4.0 / gamma)) / 2.0
    return u ** 2


def wdrcoe_eigenvalue_v(gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """Vectorised W-DRCOE covariance eigenvalues.

    Closed form: s_i = ((sqrt(b_i) + sqrt(b_i + 4/gamma)) / 2)^2
    # Custom — Context7 found no library equivalent for W-DRCOE spectral formula.
    """
    u = (np.sqrt(lambdas) + np.sqrt(lambdas + 4.0 / gamma)) / 2.0
    return u ** 2


def wise_eigenvalue(gamma: float, b: float) -> float:
    """Scalar WISE covariance eigenvalue (kept for backward compatibility)."""
    v = (-np.sqrt(gamma * b) + np.sqrt(gamma * b + 4.0)) / 2.0
    mu = v ** 2
    if mu < 1e-12:
        return 1.0 / 1e-12
    return 1.0 / mu


def wise_eigenvalue_v(gamma: float, lambdas: np.ndarray) -> np.ndarray:
    """Vectorised WISE covariance eigenvalues.

    Closed form: s_i = 1 / v_i^2 where v_i = (-sqrt(gamma*b_i) + sqrt(gamma*b_i+4))/2
    # Custom — Context7 found no library equivalent for WISE spectral formula.
    """
    v = (-np.sqrt(gamma * lambdas) + np.sqrt(gamma * lambdas + 4.0)) / 2.0
    mu = np.maximum(v ** 2, 1e-12)
    return 1.0 / mu


def _wdrcoe_constraint(gamma: float, lambdas: np.ndarray, rho: float) -> float:
    """Residual: sum_i d_W(s_DRCOE_i, lambda_i) - rho."""
    s = np.maximum(wdrcoe_eigenvalue_v(gamma, lambdas), 1e-12)
    return float(np.sum(wasserstein_div_v(s, lambdas))) - rho


def _wise_constraint(gamma: float, lambdas: np.ndarray, rho: float) -> float:
    """Residual: sum_i d_W(s_WISE_i, lambda_i) - rho."""
    s = np.maximum(wise_eigenvalue_v(gamma, lambdas), 1e-12)
    return float(np.sum(wasserstein_div_v(s, lambdas))) - rho


def wdrcoe_estimate(
    Sigma_hat: np.ndarray,
    rho: float,
) -> np.ndarray:
    """W-DRCOE: Wasserstein DRO covariance estimator (maximize log-det).

    max log det Sigma  s.t.  D_W(Sigma, Sigma_hat) <= rho

    Parameters
    ----------
    Sigma_hat : (p, p) ndarray  Nominal covariance.
    rho : float                 Ambiguity radius.

    Returns
    -------
    (p, p) ndarray  W-DRCOE covariance estimate.
    # Custom — Context7 found no library equivalent.
    """
    p = Sigma_hat.shape[0]
    lambdas, V = np.linalg.eigh(Sigma_hat)
    lambdas = np.maximum(lambdas, 1e-12)

    if rho <= 1e-14:
        return (Sigma_hat + Sigma_hat.T) / 2.0

    # Solve for gamma (Lagrange multiplier)
    gamma_hi = 1.0
    while _wdrcoe_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wdrcoe_constraint(1e-10, lambdas, rho)
    f_hi = _wdrcoe_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        # Very small rho, close to sample covariance
        s_star = lambdas
    elif f_hi > 0:
        s_star = wdrcoe_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wdrcoe_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wdrcoe_eigenvalue_v(gamma_star, lambdas)

    s_star = np.maximum(s_star, 1e-12)
    Sigma_star = V @ np.diag(s_star) @ V.T
    return (Sigma_star + Sigma_star.T) / 2.0


def wise_estimate(
    Sigma_hat: np.ndarray,
    rho: float,
) -> np.ndarray:
    """WISE: Wasserstein Inverse Covariance Estimator.

    max log det Theta  s.t.  D_W(Theta^{-1}, Sigma_hat) <= rho

    Returns the implied covariance matrix Sigma_WISE = Theta^{-1}.

    Parameters
    ----------
    Sigma_hat : (p, p) ndarray  Nominal covariance.
    rho : float                 Ambiguity radius.

    Returns
    -------
    (p, p) ndarray  WISE covariance estimate.
    # Custom — Context7 found no library equivalent.
    """
    p = Sigma_hat.shape[0]
    lambdas, V = np.linalg.eigh(Sigma_hat)
    lambdas = np.maximum(lambdas, 1e-12)

    if rho <= 1e-14:
        return (Sigma_hat + Sigma_hat.T) / 2.0

    gamma_hi = 1.0
    while _wise_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wise_constraint(1e-10, lambdas, rho)
    f_hi = _wise_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        s_star = lambdas
    elif f_hi > 0:
        s_star = wise_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wise_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wise_eigenvalue_v(gamma_star, lambdas)

    s_star = np.maximum(s_star, 1e-12)
    Sigma_star = V @ np.diag(s_star) @ V.T
    return (Sigma_star + Sigma_star.T) / 2.0


def wdrcoe_estimate_from_eig(
    lambdas: np.ndarray,
    V: np.ndarray,
    rho: float,
) -> np.ndarray:
    """W-DRCOE estimate from precomputed eigendecomposition.

    Avoids repeating ``eigh`` when sweeping rho over a grid for the same
    nominal covariance (e.g. inside ``tune_rho``).

    Parameters
    ----------
    lambdas : (p,) ascending eigenvalues (floored at 1e-12).
    V       : (p, p) eigenvector matrix from ``np.linalg.eigh``.
    rho     : ambiguity radius.

    Returns
    -------
    (p, p) ndarray  Symmetric W-DRCOE covariance estimate.
    """
    if rho <= 1e-14:
        Sigma = V @ np.diag(lambdas) @ V.T
        return (Sigma + Sigma.T) / 2.0

    gamma_hi = 1.0
    while _wdrcoe_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wdrcoe_constraint(1e-10, lambdas, rho)
    f_hi = _wdrcoe_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        s_star = lambdas.copy()
    elif f_hi > 0:
        s_star = wdrcoe_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wdrcoe_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wdrcoe_eigenvalue_v(gamma_star, lambdas)

    s_star = np.maximum(s_star, 1e-12)
    Sigma_star = V @ np.diag(s_star) @ V.T
    return (Sigma_star + Sigma_star.T) / 2.0


def wise_estimate_from_eig(
    lambdas: np.ndarray,
    V: np.ndarray,
    rho: float,
) -> np.ndarray:
    """WISE estimate from precomputed eigendecomposition.

    Avoids repeating ``eigh`` when sweeping rho over a grid for the same
    nominal covariance.

    Parameters
    ----------
    lambdas : (p,) ascending eigenvalues (floored at 1e-12).
    V       : (p, p) eigenvector matrix from ``np.linalg.eigh``.
    rho     : ambiguity radius.

    Returns
    -------
    (p, p) ndarray  Symmetric WISE covariance estimate (implied covariance).
    """
    if rho <= 1e-14:
        Sigma = V @ np.diag(lambdas) @ V.T
        return (Sigma + Sigma.T) / 2.0

    gamma_hi = 1.0
    while _wise_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wise_constraint(1e-10, lambdas, rho)
    f_hi = _wise_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        s_star = lambdas.copy()
    elif f_hi > 0:
        s_star = wise_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wise_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wise_eigenvalue_v(gamma_star, lambdas)

    s_star = np.maximum(s_star, 1e-12)
    Sigma_star = V @ np.diag(s_star) @ V.T
    return (Sigma_star + Sigma_star.T) / 2.0



# ---------------------------------------------------------------------------
# Eigenvalue-only variants (skip V @ diag @ V.T matrix reconstruction)
# Used in the fast evaluate() path where eigenvectors are shared across all
# estimators (sample, LW-L, LW-NL, SCOPE variants all preserve eigenvectors
# of the nominal sample covariance).
# ---------------------------------------------------------------------------

def scope_eigenvalues(
    lambdas: np.ndarray,
    tau: float,
    rho: float,
    divergence: "Literal['KL','W','SS']" = "W",
    rho_max: Optional[float] = None,
) -> np.ndarray:
    """SCOPE shrunk eigenvalues without matrix reconstruction.

    Equivalent to scope_estimate_from_eig() but skips V @ diag @ V.T.

    Parameters
    ----------
    lambdas   : (p,) ascending eigenvalues (floored at 1e-12).
    tau       : precision parameter p / ||Sigma_hat||_F^2.
    rho       : ambiguity radius.
    divergence: {"KL", "W", "SS"}.
    rho_max   : precomputed sum_i d(sqrt(1/tau), lambda_i); computed here when None.

    Returns
    -------
    s_star : (p,) ascending shrunk eigenvalues.
    """
    p = len(lambdas)
    scalar_target = 1.0 / np.sqrt(tau)

    if rho_max is None:
        rho_max = compute_rho_max(tau, lambdas, divergence)

    if rho >= rho_max:
        return np.full(p, scalar_target)

    gamma_star = solve_gamma_star(tau, rho, lambdas, divergence)

    if divergence == "KL":
        s_star = kl_phi_v(tau, gamma_star, lambdas)
    elif divergence == "W":
        s_star = wasserstein_phi_v(tau, gamma_star, lambdas)
    elif divergence == "SS":
        s_star = stein_phi_v(tau, gamma_star, lambdas)
    else:
        raise ValueError(f"Unknown divergence: {divergence}")

    return np.maximum(s_star, 1e-12)


def wdrcoe_eigenvalues(lambdas: np.ndarray, rho: float) -> np.ndarray:
    """W-DRCOE shrunk eigenvalues without matrix reconstruction.

    Parameters
    ----------
    lambdas : (p,) ascending eigenvalues (floored at 1e-12).
    rho     : ambiguity radius.

    Returns
    -------
    s_star : (p,) ascending shrunk eigenvalues.
    """
    if rho <= 1e-14:
        return lambdas.copy()

    gamma_hi = 1.0
    while _wdrcoe_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wdrcoe_constraint(1e-10, lambdas, rho)
    f_hi = _wdrcoe_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        s_star = lambdas.copy()
    elif f_hi > 0:
        s_star = wdrcoe_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wdrcoe_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wdrcoe_eigenvalue_v(gamma_star, lambdas)

    return np.maximum(s_star, 1e-12)


def wise_eigenvalues(lambdas: np.ndarray, rho: float) -> np.ndarray:
    """WISE shrunk covariance eigenvalues without matrix reconstruction.

    Parameters
    ----------
    lambdas : (p,) ascending eigenvalues (floored at 1e-12).
    rho     : ambiguity radius.

    Returns
    -------
    s_star : (p,) WISE covariance eigenvalues (ascending).
    """
    if rho <= 1e-14:
        return lambdas.copy()

    gamma_hi = 1.0
    while _wise_constraint(gamma_hi, lambdas, rho) > 0:
        gamma_hi *= 10.0
        if gamma_hi > 1e15:
            break

    f_lo = _wise_constraint(1e-10, lambdas, rho)
    f_hi = _wise_constraint(gamma_hi, lambdas, rho)

    if f_lo < 0:
        s_star = lambdas.copy()
    elif f_hi > 0:
        s_star = wise_eigenvalue_v(gamma_hi, lambdas)
    else:
        gamma_star = brentq(
            _wise_constraint, 1e-10, gamma_hi,
            args=(lambdas, rho), xtol=1e-10, rtol=1e-10
        )
        s_star = wise_eigenvalue_v(gamma_star, lambdas)

    return np.maximum(s_star, 1e-12)


def oracle_combined_loss_from_eig(
    s: np.ndarray,
    M_diag: np.ndarray,
    tau_star: float,
) -> float:
    """Fast oracle combined Stein-Frobenius loss from eigenvalues alone.

    Replaces oracle_combined_loss() when eigenvectors of Sigma_est are
    available and M_diag = diag(V^T Sigma_0 V) has been precomputed.
    Eliminates all matrix inversion and SVD-based log-determinant calls.

    For Sigma_est = V diag(s) V^T the loss decomposes as:

        logdet(Sigma_est) + tr(Sigma_est^{-1} Sigma_0)
            + (tau*/2) * (||Sigma_est||_F^2 - 2 tr(Sigma_est Sigma_0))
        = sum_i log(s_i) + sum_i M_diag_i / s_i
            + (tau*/2) * (sum_i s_i^2 - 2 sum_i s_i * M_diag_i)

    This identity holds whenever all estimators share eigenvectors V
    (sample, LW-L, LW-NL, SCOPE variants all preserve the eigenvectors
    of the nominal sample covariance).

    Parameters
    ----------
    s        : (p,) ascending eigenvalues of Sigma_est (must be all > 0).
    M_diag   : (p,) precomputed diagonal of V^T Sigma_0 V.
    tau_star : float, oracle tau = p / ||Sigma_0||_F^2.

    Returns
    -------
    float : loss value; np.nan if any eigenvalue is non-positive.
    """
    if np.any(s <= 0):
        return np.nan
    logdet = float(np.sum(np.log(s)))
    trace_inv = float(np.sum(M_diag / s))
    frob_term = float((tau_star / 2.0) * (np.sum(s ** 2) - 2.0 * np.sum(s * M_diag)))
    return logdet + trace_inv + frob_term
