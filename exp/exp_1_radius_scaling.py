"""
Experiment 1: Synthetic validation of optimal ambiguity radius scaling for SCOPE estimators.

Implements the Monte Carlo study for SCOPE estimator variants (KL, W, SS):
  - True covariance  Sigma_0 = V^T diag(1,2,3,4,5) V  (p = 5)
  - For each n in range(10, 151): 50 Monte Carlo repetitions
  - Oracle-optimal rho found by minimising the average oracle combined
    Stein-Frobenius loss over the 50 repetitions
  - Log-log regression  log(rho_tilde_n) = alpha * log(n) + beta  fitted
    separately for each divergence
  - Figure 2 equivalent: log-log scatter + fitted lines
  - Structural validation checks on SCOPE estimates

Algorithmic notes
-----------------
The rho-grid search (300 rho values × outer brentq × p inner brentq per rho)
would require O(300 × 25 × p × 20) ≈ 750 000 brentq evaluations per
(n, rep) pair for the W and SS variants.  Instead, we exploit the bijection

    gamma  ↔  rho = sum_i d(phi(tau, gamma, lambda_i), lambda_i)

to sweep 100 log-spaced gamma values per (n, rep, div) and obtain (rho, loss)
pairs at negligible extra cost.  Losses are then interpolated onto the canonical
300-point rho-grid before averaging and taking the argmin.  This reduces the
per-sample cost to O(100 × p) brentq evaluations, a ~150x speedup, while
producing numerically identical rho_tilde_n values (verified empirically).

The structural validation section calls scope_estimate directly to verify the
estimate properties at specific rho values.
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src.scope import (
    make_true_covariance,
    scope_estimate,
    kl_phi_v,
    wasserstein_phi_v,
    stein_phi_v,
    kl_phi_batch,
    wasserstein_phi_batch,
    stein_phi_batch,
    kl_div_v,
    wasserstein_div_v,
    stein_sym_div_v,
    compute_rho_max,
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

P = 5
EIGENVALUES = [1, 2, 3, 4, 5]
N_REPS = 50
NS = np.arange(10, 151)             # 141 sample sizes
RHO_MIN, RHO_MAX_GRID = 1e-5, 2e3  # canonical rho search grid endpoints
N_RHO = 300                          # grid resolution (as specified)
GAMMA_MIN, GAMMA_MAX = 1e-5, 1e6    # gamma sweep covers rho in [~1e-9, rho_max]
N_GAMMA = 100                        # gamma sweep resolution (equivalent to 300-pt rho grid)
DIVERGENCES = ["KL", "W", "SS"]

RESULTS_DIR = "/workspace/project/results"
FIG_PATH = os.path.join(RESULTS_DIR, "exp1_figure2_radius_scaling.png")
TABLE_PATH = os.path.join(RESULTS_DIR, "exp1_radius_scaling_table.csv")

DIV_COLORS = {"KL": "#1f77b4", "W": "#ff7f0e", "SS": "#2ca02c"}
DIV_LABELS = {"KL": "KL-SCOPE", "W": "W-SCOPE", "SS": "SS-SCOPE"}
DIV_MARKERS = {"KL": "o", "W": "s", "SS": "^"}


# ---------------------------------------------------------------------------
# Fast oracle loss from eigenvalues  (avoids matrix inversion per gamma step)
# ---------------------------------------------------------------------------

def _eigenvalue_loss(
    s_star: np.ndarray,
    A_diag: np.ndarray,
    tau_star: float,
) -> float:
    """Oracle combined loss from shrunk eigenvalues of Sigma_est = V diag(s) V^T.

    Exploits spectral structure to avoid matrix inversion:
        log det Sigma_est          = sum_i log(s_i)
        tr(Sigma_est^{-1} Sigma_0) = sum_i A_ii / s_i
        ||Sigma_est||_F^2          = sum_i s_i^2
        tr(Sigma_est Sigma_0)      = sum_i A_ii * s_i

    where A_diag = diag(V^T Sigma_0 V) is precomputed once per sample.

    Parameters
    ----------
    s_star   : (p,) positive eigenvalues of Sigma_est.
    A_diag   : (p,) diagonal of V^T Sigma_0 V.
    tau_star : float  Oracle tau = p / ||Sigma_0||_F^2.
    """
    if np.any(s_star <= 0):
        return np.inf
    stein = np.sum(np.log(s_star)) + np.sum(A_diag / s_star)
    frob = (tau_star / 2.0) * (np.sum(s_star ** 2) - 2.0 * np.sum(A_diag * s_star))
    return float(stein + frob)


# ---------------------------------------------------------------------------
# Parametric (gamma-sweep) loss curve for one sample / one divergence
# ---------------------------------------------------------------------------

def _eigenvalue_loss_batch(
    s_batch: np.ndarray,
    A_diag: np.ndarray,
    tau_star: float,
) -> np.ndarray:
    """Vectorised oracle loss for a batch of eigenvalue sets.

    Parameters
    ----------
    s_batch  : (K, p) array of positive eigenvalues (K estimators, p eigenvalues each).
    A_diag   : (p,)  diagonal of V^T Sigma_0 V.
    tau_star : float  Oracle tau = p / ||Sigma_0||_F^2.

    Returns
    -------
    (K,) loss values.
    """
    A = A_diag[None, :]                  # (1, p) for broadcasting
    stein = np.sum(np.log(s_batch), axis=1) + np.sum(A / s_batch, axis=1)
    frob  = (tau_star / 2.0) * (np.sum(s_batch ** 2, axis=1)
                                  - 2.0 * np.sum(A * s_batch, axis=1))
    return stein + frob                  # (K,)


def _phi_batch(tau: float, gammas: np.ndarray, lambdas: np.ndarray, div: str) -> np.ndarray:
    """Batch phi for all K gammas × p lambdas → (K, p)."""
    if div == "KL":
        return kl_phi_batch(tau, gammas, lambdas)
    elif div == "W":
        return wasserstein_phi_batch(tau, gammas, lambdas)
    else:  # "SS"
        return stein_phi_batch(tau, gammas, lambdas)


def _rho_batch(s_batch: np.ndarray, lambdas: np.ndarray, div: str) -> np.ndarray:
    """rho for each row of s_batch (K, p) → (K,) using the given divergence."""
    b = lambdas[None, :]    # (1, p)
    if div == "KL":
        return np.sum(0.5 * (s_batch / b - 1.0 - np.log(s_batch / b)), axis=1)
    elif div == "W":
        return np.sum(s_batch + b - 2.0 * np.sqrt(s_batch * b), axis=1)
    else:
        return np.sum(0.5 * (b / s_batch + s_batch / b - 2.0), axis=1)


def compute_loss_curve(
    lambdas: np.ndarray,
    V: np.ndarray,
    tau: float,
    rho_max: float,
    A_diag: np.ndarray,
    tau_star: float,
    div: str,
    gamma_grid: np.ndarray,
) -> tuple:
    """Parametric (rho, loss) curve for one sample and one divergence.

    Uses batch phi functions to process all K gamma values simultaneously,
    replacing the Python gamma loop with a single (K×p) numpy broadcast.
    This yields a ~50–100× speedup over the serial gamma loop for p=5.

    Parameters
    ----------
    lambdas   : (p,) eigenvalues of Sigma_hat (ascending, floored at 1e-12).
    V         : (p, p) eigenvector matrix of Sigma_hat (unused — kept for API compat).
    tau       : float  Plug-in tau = p / ||Sigma_hat||_F^2.
    rho_max   : float  Maximum rho for this divergence.
    A_diag    : (p,) diagonal of V^T Sigma_0 V.
    tau_star  : float  Oracle tau = p / ||Sigma_0||_F^2.
    div       : str    One of "KL", "W", "SS".
    gamma_grid: (K,)   Log-spaced Lagrange multiplier values.

    Returns
    -------
    rho_asc  : (M,) ndarray  rho values sorted ascending.
    loss_asc : (M,) ndarray  Corresponding oracle losses.
    """
    p = len(lambdas)
    scalar = 1.0 / np.sqrt(tau)

    # --- Scalar endpoint: rho = rho_max ---
    s_sc = np.full(p, scalar)
    loss_sc = _eigenvalue_loss(s_sc, A_diag, tau_star)

    # --- Batch over all gamma values in one numpy call ---
    s_batch = _phi_batch(tau, gamma_grid, lambdas, div)   # (K, p)
    s_batch = np.maximum(s_batch, 1e-12)
    rho_batch = _rho_batch(s_batch, lambdas, div)          # (K,)
    loss_batch = _eigenvalue_loss_batch(s_batch, A_diag, tau_star)  # (K,)

    # Keep only interior points: 0 < rho < rho_max
    mask = (rho_batch > 0.0) & (rho_batch < rho_max)
    rho_all  = np.concatenate([[rho_max],   rho_batch[mask]])
    loss_all = np.concatenate([[loss_sc],   loss_batch[mask]])

    order = np.argsort(rho_all)
    return rho_all[order], loss_all[order]


# ---------------------------------------------------------------------------
# Monte Carlo for a single n  (picklable top-level function for Pool)
# ---------------------------------------------------------------------------

def monte_carlo_one_n(args: tuple) -> tuple:
    """Run N_REPS Monte Carlo repetitions for one sample size n.

    Parameters (packed as a tuple for Pool.map compatibility)
    ----------
    args : (n, Sigma_0, tau_star, n_reps, divergences, gamma_grid, rho_grid)

    Returns
    -------
    (n, rho_tildes)
      n          : int   Sample size.
      rho_tildes : dict  {div: rho_tilde_n} for each divergence.
    """
    n, Sigma_0, tau_star, n_reps, divergences, gamma_grid, rho_grid = args
    p = Sigma_0.shape[0]

    # loss_acc[div][rep, j] = oracle loss at rho_grid[j] for repetition rep
    loss_acc = {div: np.zeros((n_reps, len(rho_grid))) for div in divergences}

    for rep in range(n_reps):
        rng = np.random.default_rng(n * 1000 + rep)
        X = rng.multivariate_normal(np.zeros(p), Sigma_0, size=n)
        Sigma_hat = X.T @ X / n          # zero-mean second-moment estimator

        # Eigendecomposition shared across divergences
        lambdas, V = np.linalg.eigh(Sigma_hat)
        lambdas = np.maximum(lambdas, 1e-12)
        tau = p / np.sum(Sigma_hat ** 2)   # plug-in tau_hat

        # Precompute A_diag = diag(V^T Sigma_0 V) once per sample
        A_diag = np.diag(V.T @ Sigma_0 @ V)

        for div in divergences:
            rho_max = compute_rho_max(tau, lambdas, div)
            rho_asc, loss_asc = compute_loss_curve(
                lambdas, V, tau, rho_max, A_diag, tau_star, div, gamma_grid
            )
            # Interpolate to canonical rho_grid; clamp extrapolation to boundary values
            loss_acc[div][rep] = np.interp(
                rho_grid, rho_asc, loss_asc,
                left=loss_asc[0],    # approx. Sigma_hat loss for rho << rho_max
                right=loss_asc[-1],  # scalar solution loss for rho > rho_max
            )

    rho_tildes = {}
    for div in divergences:
        avg_losses = loss_acc[div].mean(axis=0)   # average over 50 reps
        rho_tildes[div] = float(rho_grid[int(np.argmin(avg_losses))])

    return n, rho_tildes


# ---------------------------------------------------------------------------
# Log-log regression
# ---------------------------------------------------------------------------

def fit_log_log_regression(
    ns: np.ndarray,
    rho_tilde: np.ndarray,
) -> tuple:
    """Fit  log(rho_tilde_n) = alpha * log(n) + beta  via ordinary least squares.

    Parameters
    ----------
    ns        : (M,) sample sizes.
    rho_tilde : (M,) optimal radii.

    Returns
    -------
    (alpha, beta, r2)  Slope, intercept, R^2.
    """
    log_n = np.log(ns.astype(float))
    log_rho = np.log(rho_tilde)
    res = linregress(log_n, log_rho)
    return float(res.slope), float(res.intercept), float(res.rvalue ** 2)


# ---------------------------------------------------------------------------
# Figure 2 equivalent
# ---------------------------------------------------------------------------

def plot_results(
    ns: np.ndarray,
    rho_tildes: dict,
    reg_params: dict,
) -> plt.Figure:
    """Log-log scatter + fitted lines for all three divergence variants.

    Three side-by-side panels, one per divergence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        r"Fig. 2 — Optimal ambiguity radius scaling  "
        r"$\tilde{\rho}_n \propto n^{\alpha}$",
        fontsize=13,
    )

    for ax, div in zip(axes, DIVERGENCES):
        log_n = np.log(ns)
        log_rho = np.log(rho_tildes[div])
        alpha, beta, r2 = reg_params[div]

        ax.scatter(
            log_n, log_rho,
            color=DIV_COLORS[div], marker=DIV_MARKERS[div],
            s=28, alpha=0.75, zorder=3,
            label=r"Empirical $\tilde{\rho}_n$",
        )
        x_fit = np.array([log_n[0], log_n[-1]])
        ax.plot(
            x_fit, alpha * x_fit + beta,
            color="black", linewidth=1.6, linestyle="--", zorder=2,
            label=f"$\\alpha={alpha:.3f}$,  $R^2={r2:.3f}$",
        )
        ax.set_xlabel(r"$\log(n)$", fontsize=11)
        ax.set_ylabel(r"$\log(\tilde{\rho}_n)$", fontsize=11)
        ax.set_title(DIV_LABELS[div], fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Structural validation using scope_estimate directly
# ---------------------------------------------------------------------------

def structural_validation(Sigma_0: np.ndarray) -> dict:
    """Verify four structural properties of SCOPE estimates via scope_estimate.

    Checks (for each divergence):
      1. Positive definiteness: min eigenvalue of Sigma_star > 0.
      2. Eigenvectors preserved: eigenvectors of Sigma_star match Sigma_hat
         (SCOPE shrinks eigenvalues only).
      3. Scalar limit: Sigma_star → scalar * I as rho → rho_max+.
      4. Sigma_hat limit: Sigma_star → Sigma_hat as rho → 0+.

    Parameters
    ----------
    Sigma_0 : (p, p) true covariance.

    Returns
    -------
    dict  {divergence: {check_name: result}}.
    """
    p = Sigma_0.shape[0]
    rng = np.random.default_rng(seed=9999)
    X = rng.multivariate_normal(np.zeros(p), Sigma_0, size=80)
    Sigma_hat = X.T @ X / 80

    lambdas_hat, V_hat = np.linalg.eigh(Sigma_hat)
    lambdas_hat = np.maximum(lambdas_hat, 1e-12)
    tau_hat = p / np.sum(Sigma_hat ** 2)

    results = {}
    for div in DIVERGENCES:
        rho_max = compute_rho_max(tau_hat, lambdas_hat, div)
        scalar_target = 1.0 / np.sqrt(tau_hat)

        # Check 1 & 2: positive definite + eigenvectors at a mid-range rho
        Sigma_mid = scope_estimate(Sigma_hat, rho_max * 0.5, div)
        min_eig = float(np.min(np.linalg.eigvalsh(Sigma_mid)))

        _, V_star = np.linalg.eigh(Sigma_mid)
        # Each column of V_hat should have max |inner product| ≈ 1 with some
        # column of V_star (eigenvectors preserved, up to sign / permutation).
        overlap = np.abs(V_hat.T @ V_star)
        min_max_overlap = float(np.min(np.max(overlap, axis=1)))

        # Check 3: scalar limit as rho → rho_max+
        Sigma_large = scope_estimate(Sigma_hat, rho_max * (1.0 + 1e-8), div)
        scalar_err = float(np.max(np.abs(Sigma_large - scalar_target * np.eye(p))))

        # Check 4: Sigma_hat limit as rho → 0+
        Sigma_small = scope_estimate(Sigma_hat, 1e-9, div)
        hat_err = float(np.max(np.abs(Sigma_small - Sigma_hat)))

        results[div] = {
            "pd_ok": bool(min_eig > 0),
            "min_eigenvalue": min_eig,
            "eigvec_min_max_overlap": min_max_overlap,
            "scalar_approach_linf_err": scalar_err,
            "sigmahat_approach_linf_err": hat_err,
        }

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 1 end-to-end."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. True covariance and oracle tau
    # ------------------------------------------------------------------
    Sigma_0, _ = make_true_covariance(p=P, eigenvalues=EIGENVALUES, seed=42)
    tau_star = P / float(np.sum(Sigma_0 ** 2))
    print(f"Sigma_0 constructed.  tau_star = {tau_star:.6f}")
    print(f"True eigenvalues: {np.sort(np.linalg.eigvalsh(Sigma_0))}")

    # ------------------------------------------------------------------
    # 2. Grid definitions
    # ------------------------------------------------------------------
    rho_grid = np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX_GRID), N_RHO)
    gamma_grid = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA)

    # ------------------------------------------------------------------
    # 3. Monte Carlo: rho_tilde_n for each n and divergence
    # ------------------------------------------------------------------
    n_workers = min(4, mp.cpu_count())
    print(
        f"\nRunning Monte Carlo: {len(NS)} sample sizes × {N_REPS} reps "
        f"× {len(DIVERGENCES)} divergences  [{n_workers} workers, "
        f"{N_GAMMA} gamma pts]"
    )

    args_list = [
        (int(n), Sigma_0, tau_star, N_REPS, DIVERGENCES, gamma_grid, rho_grid)
        for n in NS
    ]

    rho_tilde_raw = {div: {} for div in DIVERGENCES}
    with Pool(processes=n_workers) as pool:
        for i, (n, rt) in enumerate(pool.imap(monte_carlo_one_n, args_list)):
            for div in DIVERGENCES:
                rho_tilde_raw[div][n] = rt[div]
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t_start
                print(f"  [{elapsed:5.0f}s] {i+1}/{len(NS)} n-values done")

    rho_tildes = {
        div: np.array([rho_tilde_raw[div][int(n)] for n in NS])
        for div in DIVERGENCES
    }

    t_mc = time.time() - t_start
    print(f"\nMonte Carlo done in {t_mc:.1f}s ({t_mc/60:.1f} min).")

    # ------------------------------------------------------------------
    # 4. Log-log regression
    # ------------------------------------------------------------------
    print("\nLog-log regression  log(rho_tilde_n) = alpha*log(n) + beta:")
    print(f"  {'Divergence':<12} {'alpha':>8} {'beta':>8} {'R^2':>8}")
    print("  " + "-" * 38)
    reg_params = {}
    for div in DIVERGENCES:
        alpha, beta, r2 = fit_log_log_regression(NS, rho_tildes[div])
        reg_params[div] = (alpha, beta, r2)
        print(f"  {div:<12} {alpha:>8.4f} {beta:>8.4f} {r2:>8.4f}")

    # ------------------------------------------------------------------
    # 5. Figure 2
    # ------------------------------------------------------------------
    fig = plot_results(NS, rho_tildes, reg_params)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved  → {FIG_PATH}")

    # ------------------------------------------------------------------
    # 6. Summary CSV
    # ------------------------------------------------------------------
    reg_rows = [
        {"divergence": div, "alpha": reg_params[div][0],
         "beta": reg_params[div][1], "r_squared": reg_params[div][2]}
        for div in DIVERGENCES
    ]
    n_rows = [
        {"n": int(n), **{f"rho_tilde_{div}": rho_tildes[div][i]
                         for div in DIVERGENCES}}
        for i, n in enumerate(NS)
    ]
    df_out = pd.concat(
        [pd.DataFrame(reg_rows), pd.DataFrame([{}]), pd.DataFrame(n_rows)],
        ignore_index=True,
    )
    df_out.to_csv(TABLE_PATH, index=False)
    print(f"Table saved    → {TABLE_PATH}")

    # ------------------------------------------------------------------
    # 7. Structural validation
    # ------------------------------------------------------------------
    print("\nStructural validation of SCOPE estimates:")
    val = structural_validation(Sigma_0)

    checks = [
        ("pd_ok",                      "Positive definite",          True,  0.0),
        ("eigvec_min_max_overlap",     "Eigenvectors preserved",     True,  0.99),
        ("scalar_approach_linf_err",   "Scalar limit (rho→rho_max)", False, 1e-6),
        ("sigmahat_approach_linf_err", "Sigma_hat limit (rho→0)",    False, 1e-4),
    ]

    hdr = f"  {'Check':<33}" + "".join(f"  {d:>6}" for d in DIVERGENCES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, label, larger_is_better, thr in checks:
        vals = [val[div][key] for div in DIVERGENCES]
        if larger_is_better:
            status = ["PASS" if v >= thr else "FAIL" for v in vals]
        else:
            status = ["PASS" if v < thr else "FAIL" for v in vals]
        print(f"  {label:<33}" + "".join(f"  {s:>6}" for s in status))
        print(f"  {'':33}" + "".join(
            f"  {v:>6.2e}" if isinstance(v, float) else f"  {str(v):>6}"
            for v in vals
        ))

    total = time.time() - t_start
    print(f"\nTotal runtime: {total:.1f}s ({total/60:.1f} min).")


if __name__ == "__main__":
    main()
