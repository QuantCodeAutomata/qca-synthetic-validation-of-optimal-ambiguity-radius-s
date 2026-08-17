"""
Experiment 3: Hyperspectral Anomaly Detection with Simulated Data.

Since real hyperspectral datasets (San Diego, Pavia, HYDICE, Texas Coast, SpecTIR)
require licenses, we simulate hyperspectral image cubes that mimic the paper's setup.

Methodology:
  - 5 simulated datasets with varying spectral dimension p (50, 100, 150, 200, 224)
  - N ≈ p pixels per dataset; 5% are anomalies
  - Background: N(0, Sigma_bg); Anomalies: N(mu_anomaly, Sigma_bg), mu_anomaly = 3*e_1
  - Eight covariance estimators feed an RX (Mahalanobis) anomaly detector
  - rho for SCOPE/DRCOE/WISE tuned via cross-validated trace proxy on 50% pixel split
  - Performance metric: AUC of RX scores against ground-truth anomaly labels

Outputs:
  - /workspace/project/results/exp3_hyperspectral_auc_table.csv
  - /workspace/project/results/exp3_auc_comparison.png
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf

from src.scope import scope_estimate, wdrcoe_estimate, wise_estimate
from src.benchmark_estimators import lw_nonlinear_shrinkage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = [
    {'name': 'SIM-p50',  'p': 50,  'N': 60,  'seed': 43},
    {'name': 'SIM-p100', 'p': 100, 'N': 110, 'seed': 44},
    {'name': 'SIM-p150', 'p': 150, 'N': 160, 'seed': 45},
    {'name': 'SIM-p200', 'p': 200, 'N': 210, 'seed': 46},
    {'name': 'SIM-p224', 'p': 224, 'N': 230, 'seed': 47},
]
ANOMALY_FRACTION = 0.05

METHODS = ['sample', 'LW-L', 'LW-NL', 'W-SCOPE', 'KL-SCOPE', 'SS-SCOPE', 'W-DRCOE', 'WISE']

# Log-spaced rho grid from 1e-5 to 200 (30 candidate values)
RHO_GRID = np.logspace(-5, np.log10(200), 30)

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_hyperspectral_data(p: int, N: int, anomaly_fraction: float, seed: int):
    """Simulate a hyperspectral image cube.

    Background covariance Sigma_bg = Q diag(1,...,p) Q^T for random orthogonal Q.
    Anomaly mean: mu_anomaly = 3 * e_1 (anomalous in first spectral band).

    Returns
    -------
    Y      : (N, p) ndarray  All pixels (background + anomaly).
    labels : (N,) ndarray    0 = background, 1 = anomaly.
    """
    rng = np.random.default_rng(seed)

    # Build Sigma_bg via QR decomposition
    G = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(R))            # sign-normalise columns
    eigenvalues = np.arange(1, p + 1, dtype=float)
    Sigma_bg = Q @ np.diag(eigenvalues) @ Q.T
    Sigma_bg = (Sigma_bg + Sigma_bg.T) / 2.0

    N_anom = max(1, int(round(N * anomaly_fraction)))
    N_bg   = N - N_anom

    mu_anomaly = np.zeros(p)
    mu_anomaly[0] = 3.0

    L = np.linalg.cholesky(Sigma_bg)
    Y_bg   = rng.standard_normal((N_bg,   p)) @ L.T
    Y_anom = rng.standard_normal((N_anom, p)) @ L.T + mu_anomaly

    Y      = np.vstack([Y_bg, Y_anom])
    labels = np.array([0] * N_bg + [1] * N_anom)
    return Y, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bessel_cov(X: np.ndarray) -> np.ndarray:
    """Centered, Bessel-corrected sample covariance from (n, p) data."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    S  = Xc.T @ Xc / (n - 1)
    return (S + S.T) / 2.0


def safe_inv(A: np.ndarray, force_pinv: bool = False) -> np.ndarray:
    """Invert A; fall back to pseudoinverse when A is not PD."""
    if force_pinv:
        return np.linalg.pinv(A)
    eigvals = np.linalg.eigvalsh(A)
    if np.all(eigvals > 0):
        return np.linalg.inv(A)
    return np.linalg.pinv(A)


def apply_rho_method(method: str, Sigma_hat: np.ndarray, rho: float, tau: float) -> np.ndarray:
    """Apply a rho-parameterised estimator to Sigma_hat."""
    div_map = {'W-SCOPE': 'W', 'KL-SCOPE': 'KL', 'SS-SCOPE': 'SS'}
    if method in div_map:
        return scope_estimate(Sigma_hat, rho, divergence=div_map[method], tau=tau)
    elif method == 'W-DRCOE':
        return wdrcoe_estimate(Sigma_hat, rho)
    elif method == 'WISE':
        return wise_estimate(Sigma_hat, rho)
    raise ValueError(f'Unknown rho-method: {method}')


# ---------------------------------------------------------------------------
# rho tuning via cross-validated trace proxy
# ---------------------------------------------------------------------------

def tune_rho(method: str, Y_centered: np.ndarray, rho_grid: np.ndarray) -> float:
    """Select rho that minimises tr(Sigma_est(rho)^{-1} @ Sigma_hat_val) on a 50% val split.

    Steps:
      1. Split pixels 50 / 50.
      2. Estimate Sigma_hat_train on the train half.
      3. For each rho, compute Sigma_est(rho) from Sigma_hat_train.
      4. Evaluate tr(Sigma_est^{-1} @ Sigma_hat_val) on the val half (proxy for fit quality).
      5. Return the rho that minimises the proxy.
    """
    N = Y_centered.shape[0]
    N_train = N // 2
    Sigma_hat_train = bessel_cov(Y_centered[:N_train])
    Sigma_hat_val   = bessel_cov(Y_centered[N_train:])

    p   = Sigma_hat_train.shape[0]
    tau = p / float(np.sum(Sigma_hat_train ** 2))

    best_rho, best_proxy = rho_grid[0], np.inf
    for rho in rho_grid:
        try:
            Sigma_est = apply_rho_method(method, Sigma_hat_train, rho, tau)
            proxy = float(np.trace(np.linalg.inv(Sigma_est) @ Sigma_hat_val))
        except Exception:
            continue
        if proxy < best_proxy:
            best_proxy = proxy
            best_rho   = rho

    return best_rho


# ---------------------------------------------------------------------------
# Covariance estimation dispatcher
# ---------------------------------------------------------------------------

def estimate_covariance(method: str, Y_centered: np.ndarray,
                        Sigma_hat: np.ndarray) -> np.ndarray:
    """Estimate the background covariance using the specified method.

    Parameters
    ----------
    method      : estimator name
    Y_centered  : (N, p) centered pixel matrix
    Sigma_hat   : Bessel-corrected sample covariance of Y_centered
    """
    p = Sigma_hat.shape[0]

    if method == 'sample':
        return Sigma_hat

    elif method == 'LW-L':
        lw = LedoitWolf(store_precision=False, assume_centered=True)
        lw.fit(Y_centered)
        C = lw.covariance_
        return (C + C.T) / 2.0

    elif method == 'LW-NL':
        return lw_nonlinear_shrinkage(Y_centered, assume_centered=True)

    else:  # rho-parameterised methods
        rho_star = tune_rho(method, Y_centered, RHO_GRID)
        tau = p / float(np.sum(Sigma_hat ** 2))
        return apply_rho_method(method, Sigma_hat, rho_star, tau)


# ---------------------------------------------------------------------------
# RX detector
# ---------------------------------------------------------------------------

def rx_scores(Y_centered: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Compute RX (Mahalanobis) anomaly scores for each pixel.

    score_i = x_i^T Sigma^{-1} x_i
    """
    return np.einsum('nd,dp,np->n', Y_centered, Sigma_inv, Y_centered)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    records = []

    for ds in DATASETS:
        name, p, N = ds['name'], ds['p'], ds['N']
        print(f'\n=== {name}: p={p}, N={N} ===')

        Y, labels = generate_hyperspectral_data(p, N, ANOMALY_FRACTION, ds['seed'])
        mu_hat    = Y.mean(axis=0)
        Y_c       = Y - mu_hat                 # centered pixels
        Sigma_hat = bessel_cov(Y_c)            # sample covariance (all pixels)

        for method in METHODS:
            try:
                Sigma_est = estimate_covariance(method, Y_c, Sigma_hat)
                # Sample covariance may be near-singular (N ≈ p); use pinv
                use_pinv  = (method == 'sample')
                Sigma_inv = safe_inv(Sigma_est, force_pinv=use_pinv)
                scores    = rx_scores(Y_c, Sigma_inv)
                auc       = roc_auc_score(labels, scores)
            except Exception as exc:
                print(f'  {method}: ERROR – {exc}')
                auc = float('nan')

            print(f'  {method}: AUC = {auc:.4f}')
            records.append({'Dataset': name, 'p': p, 'N': N, 'Method': method, 'AUC': auc})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame):
    pivot = df.pivot_table(index='Method', columns='Dataset', values='AUC')
    pivot = pivot.reindex(index=METHODS)
    pivot = pivot.reindex(columns=[d['name'] for d in DATASETS])
    pivot.to_csv(os.path.join(RESULTS_DIR, 'exp3_hyperspectral_auc_table.csv'))
    print('\n=== AUC Table (rows=methods, cols=datasets) ===')
    print(pivot.round(4).to_string())
    return pivot


def save_figure(df: pd.DataFrame):
    pivot = df.pivot_table(index='Method', columns='Dataset', values='AUC')
    pivot = pivot.reindex(index=METHODS)
    pivot = pivot.reindex(columns=[d['name'] for d in DATASETS])

    p_values = [d['p'] for d in DATASETS]
    colors   = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(METHODS):
        aucs = [pivot.loc[method, d['name']] for d in DATASETS]
        ax.plot(p_values, aucs, marker='o', label=method, color=colors[i])

    ax.set_xlabel('Spectral dimension  p', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Exp 3 – Hyperspectral Anomaly Detection (Simulated)\nRX Detector AUC vs Spectral Dimension', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'exp3_auc_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Figure saved → {RESULTS_DIR}/exp3_auc_comparison.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df = run_experiment()
    save_table(df)
    save_figure(df)
    print('\nDone.')
