"""
Experiment 4: A/B Test Metric Learning.

Learns a direction w that distinguishes A/B treatment pairs from A/A null pairs
by aggregating per-pair inverse-covariance weighted mean differences over training
pairs, then scoring held-out test pairs with a normalised z-statistic.

Data Generating Process (documented assumption):
  - V = Q from QR(G), G ~ N(0,1)^{50×50} with seed=42, sign-normalised
  - Sigma_1 = V^T diag(1,...,50) V / 50    (control group covariance)
  - Sigma_2 = V^T diag(2,...,51) V / 50    (treatment group covariance)
  - mu = 0.3 * ones(50) / sqrt(50)          (unit-norm treatment effect × 0.3)
  - A/A null : both groups from N(0, Sigma_1)
  - A/B pairs: A from N(0, Sigma_1), B from N(mu, Sigma_2)

Parameters:
  - p = 50 features
  - 20 repetitions
  - Training: 100 pairs (60 A/B + 40 A/A); n_train ∈ {50,60,70,100,120} obs/group
  - Testing:  200 pairs (120 A/B + 80 A/A); 200 obs/group
  - Seed = rep * 10_000 + n  for each (rep, n) combination

Methods: sample, LW-L, LW-NL, W-SCOPE, KL-SCOPE, SS-SCOPE, W-DRCOE
  - SCOPE / DRCOE: fixed rho = 0.5; tau = p / ||Sigma_hat||_F^2 per group
  - For f(Sigma_hat) with 'sample': return Sigma_hat unchanged.
  - Matrix inversion for w_k: np.linalg.pinv for numerical stability.

Outputs:
  - /workspace/project/results/exp4_ab_testing_auc_table.csv
  - /workspace/project/results/exp4_auc_vs_n.png
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from numba import njit

from src.scope import wdrcoe_estimate
from src.benchmark_estimators import lw_nonlinear_shrinkage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

P = 50
N_REPS = 20
N_TRAIN_PAIRS = 100
N_AB_TRAIN    = 60     # A/B pairs in training
N_AA_TRAIN    = 40     # A/A pairs in training
N_TEST_PAIRS  = 200
N_AB_TEST     = 120
N_AA_TEST     = 80
N_TEST_OBS    = 200    # observations per group in test pairs

N_VALUES = [50, 60, 70, 100, 120]

METHODS = ['sample', 'LW-L', 'LW-NL', 'W-SCOPE', 'KL-SCOPE', 'SS-SCOPE', 'W-DRCOE']

SCOPE_RHO = 0.5        # fixed rho for SCOPE / DRCOE

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# True covariance construction (seed=42, sign-normalised QR)
# ---------------------------------------------------------------------------

def build_true_covariances(p: int = P):
    """Build Sigma_1, Sigma_2 and mu from the documented DGP.

    Sigma_k = V^T diag(lambda_k) V / p  where lambda_1 = [1,...,p],
    lambda_2 = [2,...,p+1], and V comes from QR(G) with seed=42.
    """
    rng = np.random.default_rng(42)
    G   = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(G)
    Q    = Q * np.sign(np.diag(R))    # sign-normalise (V = Q)
    V    = Q

    lam1   = np.arange(1, p + 1, dtype=float)
    lam2   = np.arange(2, p + 2, dtype=float)
    Sigma1 = V.T @ np.diag(lam1 / p) @ V
    Sigma2 = V.T @ np.diag(lam2 / p) @ V
    Sigma1 = (Sigma1 + Sigma1.T) / 2.0
    Sigma2 = (Sigma2 + Sigma2.T) / 2.0

    mu = 0.3 * np.ones(p) / np.sqrt(p)
    return Sigma1, Sigma2, mu


# ---------------------------------------------------------------------------
# Covariance estimator applied to group data
# ---------------------------------------------------------------------------

def sample_cov_centered(X: np.ndarray) -> np.ndarray:
    """Bessel-corrected sample covariance of (n, p) data."""
    n  = X.shape[0]
    Xc = X - X.mean(axis=0)
    S  = Xc.T @ Xc / (n - 1)
    return (S + S.T) / 2.0


# ---------------------------------------------------------------------------
# Fast SCOPE phi solvers – JIT-compiled Newton loops (no per-eigenvalue brentq)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _w_phi_v(tau: float, gamma: float, lam: np.ndarray) -> np.ndarray:
    """JIT-compiled Wasserstein phi via Newton on u = sqrt(a).

    Solves element-wise: tau*u^4 + gamma*u^2 - gamma*sqrt(b)*u - 1 = 0.
    Compiled with numba; ~100x faster than a Python for-loop.
    """
    p   = lam.shape[0]
    out = np.empty(p)
    for j in range(p):
        b    = lam[j]
        sb   = b ** 0.5
        u    = sb
        for _ in range(50):
            u2 = u * u
            f  = tau * u2 * u2 + gamma * u2 - gamma * sb * u - 1.0
            fp = 4.0 * tau * u2 * u + 2.0 * gamma * u - gamma * sb
            if abs(fp) < 1e-15:
                fp = 1e-15
            du = f / fp
            # Limit step
            if du > 0.9 * u:
                du = 0.9 * u
            u = u - du
            if u < 1e-12:
                u = 1e-12
            if abs(du) < 1e-10:
                break
        out[j] = u * u
    return out


@njit(cache=True)
def _ss_phi_v(tau: float, gamma: float, lam: np.ndarray) -> np.ndarray:
    """JIT-compiled SS phi via Newton.

    Solves element-wise: 2*tau*b*a^3 + gamma*a^2 - 2*b*a - gamma*b^2 = 0.
    """
    p   = lam.shape[0]
    out = np.empty(p)
    for j in range(p):
        b  = lam[j]
        a  = b
        for _ in range(50):
            a2 = a * a
            f  = 2.0 * tau * b * a2 * a + gamma * a2 - 2.0 * b * a - gamma * b * b
            fp = 6.0 * tau * b * a2 + 2.0 * gamma * a - 2.0 * b
            if abs(fp) < 1e-15:
                fp = 1e-15
            da = f / fp
            if da > 0.9 * a:
                da = 0.9 * a
            a = a - da
            if a < 1e-12:
                a = 1e-12
            if abs(da) < 1e-10:
                break
        out[j] = a
    return out


def _kl_phi_v_safe(tau: float, gamma: float, lam: np.ndarray) -> np.ndarray:
    """Closed-form KL phi with guard against near-zero eigenvalues."""
    lam_s = np.maximum(lam, 1e-12)
    num   = -gamma + np.sqrt(np.maximum(
        gamma ** 2 + 8.0 * tau * (gamma + 2.0) * lam_s ** 2, 0.0))
    return np.maximum(num / (4.0 * tau * lam_s), 1e-12)


def _fast_scope(Sigma_hat: np.ndarray, rho: float, divergence: str,
                tau: float) -> np.ndarray:
    """Fast SCOPE via vectorised Newton phi (no per-eigenvalue brentq).

    Drop-in replacement for src.scope.scope_estimate for W, KL, and SS
    divergences.  The outer gamma_star search still uses brentq (one call),
    but every evaluation of the constraint uses an O(p) numpy operation
    instead of p nested brentq calls — roughly 50-100× faster for p=50.
    """
    p = Sigma_hat.shape[0]
    lam, V = np.linalg.eigh(Sigma_hat)
    lam    = np.maximum(lam, 1e-12)

    scalar = 1.0 / np.sqrt(tau)

    # Divergence-specific phi and divergence functions
    if divergence == 'W':
        phi_v  = _w_phi_v
        div_v  = lambda a, b: a + b - 2.0 * np.sqrt(a * b)
    elif divergence == 'SS':
        phi_v  = _ss_phi_v
        div_v  = lambda a, b: 0.5 * (b / np.maximum(a, 1e-12) +
                                       a / np.maximum(b, 1e-12) - 2.0)
    elif divergence == 'KL':
        phi_v  = _kl_phi_v_safe
        div_v  = lambda a, b: 0.5 * (a / np.maximum(b, 1e-12) - 1.0
                                      - np.log(np.maximum(a / np.maximum(b, 1e-12),
                                                           1e-300)))
    else:
        raise ValueError(f'Unknown divergence: {divergence}')

    rho_max = float(np.sum(div_v(scalar, lam)))
    if rho >= rho_max:
        return scalar * np.eye(p)

    def gamma_eq(gamma: float) -> float:
        phi = phi_v(tau, gamma, lam)
        return float(np.sum(div_v(phi, lam))) - rho

    # Expand bracket until sign change
    g_hi = 1.0
    while gamma_eq(g_hi) > 0 and g_hi < 1e14:
        g_hi *= 10.0

    if gamma_eq(0.0) < 0:
        gamma_star = 0.0
    elif gamma_eq(g_hi) > 0:
        gamma_star = g_hi
    else:
        gamma_star = brentq(gamma_eq, 0.0, g_hi, xtol=1e-8, rtol=1e-6)

    s_star = np.maximum(phi_v(tau, gamma_star, lam), 1e-12)
    Sigma_star = V @ np.diag(s_star) @ V.T
    return (Sigma_star + Sigma_star.T) / 2.0


def apply_method_to_data(method: str, X: np.ndarray) -> np.ndarray:
    """Estimate covariance of group X using the specified method.

    For SCOPE / DRCOE: compute sample covariance first, then apply spectral
    shrinkage with fixed rho = SCOPE_RHO and tau = p / ||Sigma_hat||_F^2.
    """
    p         = X.shape[1]
    Sigma_hat = sample_cov_centered(X)

    if method == 'sample':
        return Sigma_hat

    elif method == 'LW-L':
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        C = lw.covariance_
        return (C + C.T) / 2.0

    elif method == 'LW-NL':
        return lw_nonlinear_shrinkage(X, assume_centered=False)

    else:
        tau      = p / float(np.sum(Sigma_hat ** 2))
        div_map  = {'W-SCOPE': 'W', 'KL-SCOPE': 'KL', 'SS-SCOPE': 'SS'}
        if method in div_map:
            return _fast_scope(Sigma_hat, SCOPE_RHO, div_map[method], tau)
        elif method == 'W-DRCOE':
            return wdrcoe_estimate(Sigma_hat, SCOPE_RHO)
        raise ValueError(f'Unknown method: {method}')


# ---------------------------------------------------------------------------
# Core logic: weight vector and z-score
# ---------------------------------------------------------------------------

def compute_w_star(method: str, train_pairs: list, rng: np.random.Generator) -> np.ndarray:
    """Aggregate per-pair weight vectors into w_star.

    For each pair k:
      1. Estimate f(Sigma_Ak) and f(Sigma_Bk) using the method.
      2. w_k = pinv(f(Sigma_Ak) + f(Sigma_Bk)) @ (mu_hat_Bk - mu_hat_Ak).
      3. Normalise: w_k /= ||w_k|| + 1e-10.
    Return mean over all K pairs.
    """
    w_list = []
    for (X_A, X_B) in train_pairs:
        f_A = apply_method_to_data(method, X_A)
        f_B = apply_method_to_data(method, X_B)

        mu_A  = X_A.mean(axis=0)
        mu_B  = X_B.mean(axis=0)
        delta = mu_B - mu_A

        S_sum = f_A + f_B
        w_k   = np.linalg.pinv(S_sum) @ delta
        w_k   = w_k / (np.linalg.norm(w_k) + 1e-10)
        w_list.append(w_k)

    return np.mean(w_list, axis=0)


def z_score(w: np.ndarray, X_A: np.ndarray, X_B: np.ndarray) -> float:
    """Compute z-score for a test pair using sample covariances in the denominator.

    z = w^T (mu_hat_B - mu_hat_A) /
        sqrt( w^T Sigma_hat_A w / n_A + w^T Sigma_hat_B w / n_B )

    Uses sample covariances (method-agnostic) for the denominator, as specified.
    """
    n_A, n_B   = X_A.shape[0], X_B.shape[0]
    mu_A, mu_B = X_A.mean(axis=0), X_B.mean(axis=0)
    S_A = sample_cov_centered(X_A)
    S_B = sample_cov_centered(X_B)

    numerator = w @ (mu_B - mu_A)
    var_A     = w @ S_A @ w / n_A
    var_B     = w @ S_B @ w / n_B
    denom     = np.sqrt(max(var_A + var_B, 1e-12))
    return float(numerator / denom)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_obs(n: int, mu: np.ndarray, Sigma: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Draw n observations from N(mu, Sigma)."""
    L = np.linalg.cholesky(Sigma)
    return rng.standard_normal((n, mu.shape[0])) @ L.T + mu


def generate_pairs(n_obs: int, n_ab: int, n_aa: int,
                   Sigma1: np.ndarray, Sigma2: np.ndarray, mu: np.ndarray,
                   rng: np.random.Generator) -> tuple:
    """Generate A/B and A/A pairs; return (pairs list, labels array).

    Each pair is a tuple (X_A, X_B), each of shape (n_obs, p).
    Label 1 = A/B, 0 = A/A.
    """
    p      = mu.shape[0]
    pairs  = []
    labels = []
    zeros  = np.zeros(p)

    for _ in range(n_ab):
        X_A = generate_obs(n_obs, zeros, Sigma1, rng)
        X_B = generate_obs(n_obs, mu,    Sigma2, rng)
        pairs.append((X_A, X_B))
        labels.append(1)

    for _ in range(n_aa):
        X_A = generate_obs(n_obs, zeros, Sigma1, rng)
        X_B = generate_obs(n_obs, zeros, Sigma1, rng)
        pairs.append((X_A, X_B))
        labels.append(0)

    return pairs, np.array(labels)


# ---------------------------------------------------------------------------
# Single (rep, n) evaluation
# ---------------------------------------------------------------------------

def evaluate_one(rep: int, n_train: int,
                 Sigma1: np.ndarray, Sigma2: np.ndarray,
                 mu: np.ndarray) -> dict:
    """Run all methods for one (repetition, n_train) combination.

    Returns a dict {method: AUC}.
    """
    seed = rep * 10_000 + n_train
    rng  = np.random.default_rng(seed)

    # Generate training pairs
    train_pairs, _ = generate_pairs(
        n_train, N_AB_TRAIN, N_AA_TRAIN, Sigma1, Sigma2, mu, rng)

    # Generate test pairs (200 obs/group, fixed across methods)
    test_pairs, test_labels = generate_pairs(
        N_TEST_OBS, N_AB_TEST, N_AA_TEST, Sigma1, Sigma2, mu, rng)

    results = {}
    for method in METHODS:
        try:
            w_star = compute_w_star(method, train_pairs, rng)

            z_scores = np.array([z_score(w_star, xA, xB) for xA, xB in test_pairs])
            auc      = roc_auc_score(test_labels, z_scores)
        except Exception as exc:
            print(f'    rep={rep}, n={n_train}, method={method}: ERROR – {exc}')
            auc = float('nan')

        results[method] = auc
    return results


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    import time as _time
    Sigma1, Sigma2, mu = build_true_covariances(P)

    records = []
    for n in N_VALUES:
        print(f'\n=== n_train = {n} ===')
        auc_per_method = {m: [] for m in METHODS}
        t_n = _time.time()

        for rep in range(1, N_REPS + 1):
            row = evaluate_one(rep, n, Sigma1, Sigma2, mu)
            for m in METHODS:
                auc_per_method[m].append(row[m])
            if rep % 5 == 0:
                elapsed = _time.time() - t_n
                print(f'  rep {rep}/{N_REPS} done  ({elapsed:.1f}s so far)')

        for m in METHODS:
            vals = [v for v in auc_per_method[m] if not np.isnan(v)]
            mean_auc = float(np.mean(vals)) if vals else float('nan')
            std_auc  = float(np.std(vals))  if vals else float('nan')
            print(f'  {m}: mean AUC = {mean_auc:.4f} ± {std_auc:.4f}')
            records.append({
                'n_train': n,
                'Method':  m,
                'AUC_mean': mean_auc,
                'AUC_std':  std_auc,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame):
    pivot = df.pivot_table(index='Method', columns='n_train', values='AUC_mean')
    pivot = pivot.reindex(index=METHODS)
    pivot.to_csv(os.path.join(RESULTS_DIR, 'exp4_ab_testing_auc_table.csv'))
    print('\n=== Mean AUC Table (rows=methods, cols=n_train) ===')
    print(pivot.round(4).to_string())
    return pivot


def save_figure(df: pd.DataFrame):
    colors  = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(METHODS):
        sub = df[df['Method'] == method].sort_values('n_train')
        ax.plot(sub['n_train'], sub['AUC_mean'],
                marker=markers[i % len(markers)], label=method,
                color=colors[i])
        ax.fill_between(
            sub['n_train'],
            sub['AUC_mean'] - sub['AUC_std'],
            sub['AUC_mean'] + sub['AUC_std'],
            alpha=0.12, color=colors[i])

    ax.set_xlabel('Training observations per group  n', fontsize=12)
    ax.set_ylabel('Mean AUC  (20 repetitions)', fontsize=12)
    ax.set_title('Exp 4 – A/B Test Metric Learning\nMean AUC vs Training Sample Size', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp4_auc_vs_n.png'), dpi=150)
    plt.close(fig)
    print(f'Figure saved → {RESULTS_DIR}/exp4_auc_vs_n.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Warm up numba JIT (first call triggers compilation; subsequent calls are fast)
    print('Warming up numba JIT (first call compiles)...')
    _dummy = np.ones(P, dtype=float) * 0.1
    _w_phi_v(1.0, 1.0, _dummy)
    _ss_phi_v(1.0, 1.0, _dummy)
    print('JIT warm-up done.')

    df = run_experiment()
    save_table(df)
    save_figure(df)
    print('\nDone.')
