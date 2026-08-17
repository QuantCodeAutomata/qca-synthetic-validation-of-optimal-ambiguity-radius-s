"""
Experiment 5: Minimum-variance portfolio backtest with Fama-French 48 industries.

Methodology:
  - Rolling 60-month window covariance estimation
  - Tuning period: Jan 1986 – Dec 1995
  - Test period:   Jan 1996 – Aug 2025 (or latest available)
  - Methods: sample, LW-L, LW-NL, W-SCOPE, KL-SCOPE, SS-SCOPE, W-DRCOE, WISE
  - Metrics: cumulative return, annualised return, Sharpe ratio, Sortino ratio
"""

import sys
sys.path.insert(0, '/workspace/project')

import io
import os
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.optimize import minimize
try:
    import quadprog as _quadprog  # Goldfarb-Idnani active-set QP — ~0.1 ms/solve at p=48
    _HAS_QUADPROG = True
except ImportError:
    _HAS_QUADPROG = False
from sklearn.covariance import LedoitWolf

from src.benchmark_estimators import lw_nonlinear_shrinkage
from src.scope import (
    scope_estimate,
    scope_estimate_from_eig,
    wdrcoe_estimate,
    wdrcoe_estimate_from_eig,
    wise_estimate,
    wise_estimate_from_eig,
    compute_rho_max,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
       '48_Industry_Portfolios_CSV.zip')
DATA_PATH = '/workspace/project/data/ff48_returns.csv'
FIG_PATH = '/workspace/project/results/exp5_portfolio_cumulative_returns.png'
TABLE_PATH = '/workspace/project/results/exp5_portfolio_metrics_table.csv'

WINDOW = 60
RHO_GRID = np.logspace(-3, np.log10(5e3), 25)  # 25-point log grid (fast tuning)
TUNING_START = '1986-01-01'
TUNING_END = '1995-12-31'
TEST_START = '1996-01-01'
TEST_END = '2025-08-31'

METHODS_NO_RHO = ['sample', 'LW-L', 'LW-NL']
METHODS_WITH_RHO = ['W-SCOPE', 'KL-SCOPE', 'SS-SCOPE', 'W-DRCOE', 'WISE']
ALL_METHODS = METHODS_NO_RHO + METHODS_WITH_RHO

# ---------------------------------------------------------------------------
# Data download and parsing
# ---------------------------------------------------------------------------

def _parse_ff48_content(content: str) -> pd.DataFrame:
    """Parse raw text of the FF48 CSV into a monthly-returns DataFrame.

    The file is comma-delimited with multiple sections.  The first section
    contains monthly value-weighted returns.  Data rows begin with a 6-digit
    YYYYMM date token followed by 48 comma-separated return values.
    """
    lines = content.replace('\r', '').split('\n')

    # Locate the header line (starts with comma, e.g. ',Agric,Food,...')
    # and the first data row (starts with 6-digit YYYYMM date)
    header_idx = None
    first_data_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Header: first field is empty (leading comma) + >= 20 industry names
        if stripped.startswith(',') and stripped.count(',') >= 20 and header_idx is None:
            header_idx = i
        # Data rows: first comma-delimited token is a 6-digit integer date
        if header_idx is not None and first_data_idx is None:
            tok0 = stripped.split(',')[0].strip()
            if len(tok0) == 6 and tok0.isdigit():
                try:
                    date_val = int(tok0)
                    if 190000 <= date_val <= 210000:
                        first_data_idx = i
                except ValueError:
                    pass

    if first_data_idx is None:
        raise ValueError("FF48: could not locate data rows in file")

    # Parse column names from header line
    if header_idx is not None:
        col_parts = lines[header_idx].split(',')
        col_names = [c.strip() for c in col_parts[1:] if c.strip()]
    else:
        col_names = None

    # Collect the first contiguous block of data rows (stop at blank line)
    data_rows = []
    for i in range(first_data_idx, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            break
        parts = stripped.split(',')
        tok0 = parts[0].strip()
        if len(tok0) == 6 and tok0.isdigit():
            data_rows.append(parts)
        else:
            break

    if not data_rows:
        raise ValueError("FF48: no data rows parsed")

    n_assets = max(len(r) - 1 for r in data_rows)
    if col_names is None or len(col_names) < n_assets:
        col_names = [f'I{k + 1:02d}' for k in range(n_assets)]
    else:
        col_names = col_names[:n_assets]

    dates, values = [], []
    for row in data_rows:
        ds = row[0].strip()
        dates.append(
            pd.Timestamp(f'{ds[:4]}-{ds[4:6]}-01') + pd.offsets.MonthEnd(0)
        )
        vals = []
        for v in row[1: n_assets + 1]:
            try:
                vals.append(float(v.strip()))
            except ValueError:
                vals.append(np.nan)
        # Pad short rows with NaN
        while len(vals) < n_assets:
            vals.append(np.nan)
        values.append(vals)

    df = pd.DataFrame(values, index=dates, columns=col_names)
    df.index.name = 'date'
    df = df.replace(-99.99, np.nan).replace(-999.0, np.nan)
    df = df.dropna()          # keep only months where all 48 industries present
    df = df / 100.0           # percent → decimal
    return df


def download_and_parse_ff48(url: str = URL) -> pd.DataFrame:
    """Download the FF48 zip and return a cleaned monthly-returns DataFrame."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    csv_name = next(n for n in zf.namelist() if n.lower().endswith('.csv'))
    content = zf.read(csv_name).decode('utf-8', errors='replace')
    return _parse_ff48_content(content)


def load_data() -> pd.DataFrame:
    """Return FF48 monthly returns, using a cached CSV when available."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = download_and_parse_ff48()
    df.to_csv(DATA_PATH)
    return df


# ---------------------------------------------------------------------------
# Covariance estimation for financial returns
# ---------------------------------------------------------------------------

def estimate_cov_financial(
    X: np.ndarray | None,
    method: str,
    rho: float = 0.0,
    Sigma_nom: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate covariance from an (n, p) window of returns.

    Parameters
    ----------
    X        : raw return window (n, p). Required for LW-L and LW-NL.
               May be None when Sigma_nom is supplied and method is SCOPE/DRCOE/WISE.
    method   : one of 'sample', 'LW-L', 'LW-NL', 'W-SCOPE', 'KL-SCOPE',
               'SS-SCOPE', 'W-DRCOE', 'WISE'.
    rho      : ambiguity radius for methods that require it.
    Sigma_nom: precomputed centered Bessel-corrected nominal covariance.
               When provided, avoids recomputing it from X.
    """
    if Sigma_nom is None:
        assert X is not None, "Either X or Sigma_nom must be supplied."
        n, p = X.shape
        mu = X.mean(axis=0)
        Xc = X - mu
        Sigma_hat = (Xc.T @ Xc) / (n - 1)
        Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
    else:
        Sigma_hat = Sigma_nom
        p = Sigma_hat.shape[0]

    if method == 'sample':
        return Sigma_hat

    if method == 'LW-L':
        assert X is not None, "LW-L requires raw data X."
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        cov = lw.covariance_
        return (cov + cov.T) / 2.0

    if method == 'LW-NL':
        assert X is not None, "LW-NL requires raw data X."
        return lw_nonlinear_shrinkage(X, assume_centered=False)

    if method in ('W-SCOPE', 'KL-SCOPE', 'SS-SCOPE'):
        div_map = {'W-SCOPE': 'W', 'KL-SCOPE': 'KL', 'SS-SCOPE': 'SS'}
        tau = p / float(np.sum(Sigma_hat ** 2))
        return scope_estimate(Sigma_hat, rho, divergence=div_map[method], tau=tau)

    if method == 'W-DRCOE':
        return wdrcoe_estimate(Sigma_hat, rho)

    if method == 'WISE':
        return wise_estimate(Sigma_hat, rho)

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Portfolio optimisation
# ---------------------------------------------------------------------------

def solve_minvar_portfolio(Sigma: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance portfolio.

    Primary: Goldfarb-Idnani active-set QP via quadprog (~0.1 ms/solve at p=48).
    Fallback: scipy SLSQP if quadprog is unavailable or raises.

    Parameters
    ----------
    Sigma : (p, p) ndarray  Symmetric positive-definite covariance estimate.

    Returns
    -------
    (p,) ndarray  Long-only portfolio weights summing to 1.
    """
    p = Sigma.shape[0]

    if _HAS_QUADPROG:
        try:
            # quadprog.solve_qp minimises 0.5 x^T G x - a^T x
            # subject to C^T x >= b, with the first meq constraints as equalities.
            # Our problem:  min w^T Sigma w  s.t.  sum(w)=1, w>=0
            # Map: G = 2*Sigma (PD after nudge), a=0, C = [1 | I_p], b = [1, 0…0], meq=1
            G = np.asfortranarray(2.0 * Sigma + np.eye(p) * 1e-8, dtype=np.float64)
            a = np.zeros(p, dtype=np.float64)
            # C is (p, 1+p); first column is equality constraint, rest are w_i >= 0
            C = np.asfortranarray(
                np.column_stack([np.ones(p), np.eye(p)]), dtype=np.float64
            )
            b = np.concatenate([[1.0], np.zeros(p)])
            w = _quadprog.solve_qp(G, a, C, b, 1)[0]
            return np.clip(w, 0.0, None)
        except Exception:
            pass  # fall through to SLSQP

    # SLSQP fallback
    w0 = np.ones(p) / p
    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=w0,
        jac=lambda w: 2.0 * Sigma @ w,
        method='SLSQP',
        bounds=[(0.0, None)] * p,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                     'jac': lambda w: np.ones(p)},
        options={'ftol': 1e-9, 'maxiter': 500},
    )
    if res.success:
        return np.clip(res.x, 0.0, None)
    return np.ones(p) / p


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_rho(
    df: pd.DataFrame,
    tuning_dates: pd.DatetimeIndex,
    method: str,
    rho_grid: np.ndarray,
    window: int,
) -> float:
    """Return rho that maximises average portfolio return over the tuning period.

    Optimised: eigendecomposition (and tau / rho_max for SCOPE methods) is
    precomputed once per month, then the spectral-shrinkage step is swept
    across all rho values using the *_from_eig helpers — eliminating 25 ×
    |tuning_months| redundant ``eigh`` calls.
    """
    all_dates = df.index
    avg_returns = np.full(len(rho_grid), -np.inf)

    div_map = {'W-SCOPE': 'W', 'KL-SCOPE': 'KL', 'SS-SCOPE': 'SS'}
    is_scope = method in div_map
    is_drcoe = method == 'W-DRCOE'
    is_wise  = method == 'WISE'

    # ------------------------------------------------------------------
    # Phase 1: precompute everything that is rho-independent for each month
    # ------------------------------------------------------------------
    # Each entry: (lambdas, V, tau, rho_max | None, R_t)
    monthly_cache: list[tuple] = []
    for t in tuning_dates:
        t_pos = all_dates.get_loc(t)
        if t_pos < window:
            continue
        X = df.iloc[t_pos - window: t_pos].values
        R_t = df.iloc[t_pos].values

        # Centred, Bessel-corrected covariance
        Sigma_nom = np.cov(X, rowvar=False)
        Sigma_nom = (Sigma_nom + Sigma_nom.T) / 2.0

        lambdas, V = np.linalg.eigh(Sigma_nom)
        lambdas = np.maximum(lambdas, 1e-12)
        p = len(lambdas)

        if is_scope:
            divergence = div_map[method]
            tau = p / float(np.sum(Sigma_nom ** 2))
            rho_max = compute_rho_max(tau, lambdas, divergence)
            monthly_cache.append((lambdas, V, tau, rho_max, R_t))
        else:
            # W-DRCOE / WISE: tau and rho_max not needed
            monthly_cache.append((lambdas, V, None, None, R_t))

    # ------------------------------------------------------------------
    # Phase 2: sweep rho grid using precomputed eigendata
    # ------------------------------------------------------------------
    for j, rho in enumerate(rho_grid):
        monthly_rets: list[float] = []
        for entry in monthly_cache:
            lambdas, V, tau, rho_max, R_t = entry
            try:
                if is_scope:
                    divergence = div_map[method]
                    Sigma = scope_estimate_from_eig(
                        lambdas, V, tau, rho, divergence, rho_max=rho_max
                    )
                elif is_drcoe:
                    Sigma = wdrcoe_estimate_from_eig(lambdas, V, rho)
                elif is_wise:
                    Sigma = wise_estimate_from_eig(lambdas, V, rho)
                else:
                    raise ValueError(f"Unexpected method in tune_rho: {method}")
                w = solve_minvar_portfolio(Sigma)
                monthly_rets.append(float(np.dot(w, R_t)))
            except Exception:
                pass
        if monthly_rets:
            avg_returns[j] = float(np.mean(monthly_rets))

    best_idx = int(np.argmax(avg_returns))
    return float(rho_grid[best_idx])


# ---------------------------------------------------------------------------
# Out-of-sample backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    method_rho: dict,
    window: int,
) -> pd.DataFrame:
    """Run the OOS backtest for all methods.

    For each month t in `test_dates`, estimates the covariance from the
    preceding `window` months, solves the min-var portfolio, and records
    the realised return for month t.

    Returns
    -------
    pd.DataFrame  Columns = methods, rows = test months, values = returns.
    """
    all_dates = df.index
    series_dict: dict = {m: {} for m in method_rho}

    for t in test_dates:
        t_pos = all_dates.get_loc(t)
        if t_pos < window:
            continue
        X = df.iloc[t_pos - window: t_pos].values
        R_t = df.iloc[t_pos].values

        # Precompute nominal covariance once; share across all methods.
        mu = X.mean(axis=0)
        Xc = X - mu
        Sigma_nom = (Xc.T @ Xc) / (len(X) - 1)
        Sigma_nom = (Sigma_nom + Sigma_nom.T) / 2.0

        for method, rho in method_rho.items():
            try:
                if method in ('LW-L', 'LW-NL'):
                    Sigma = estimate_cov_financial(X, method, rho)
                else:
                    Sigma = estimate_cov_financial(None, method, rho,
                                                   Sigma_nom=Sigma_nom)
                w = solve_minvar_portfolio(Sigma)
                series_dict[method][t] = float(np.dot(w, R_t))
            except Exception:
                pass

    return pd.DataFrame(
        {m: pd.Series(series_dict[m]) for m in method_rho}
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series) -> dict:
    """Annualised performance metrics from a series of monthly returns."""
    r = returns.dropna().values
    if len(r) == 0:
        return {k: np.nan for k in
                ('Cumulative Return', 'Ann. Return', 'Sharpe', 'Sortino')}

    cum_ret = float(np.prod(1.0 + r) - 1.0)
    ann_ret = float(np.mean(r) * 12)

    std_r = float(np.std(r, ddof=1))
    sharpe = float((np.mean(r) / std_r) * np.sqrt(12)) if std_r > 0 else np.nan

    r_neg = r[r < 0]
    downside_std = float(np.std(r_neg, ddof=1)) if len(r_neg) > 1 else 0.0
    sortino = (float(np.mean(r) * 12) / (downside_std * np.sqrt(12))
               if downside_std > 0 else np.nan)

    return {
        'Cumulative Return': cum_ret,
        'Ann. Return': ann_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
    }


# ---------------------------------------------------------------------------
# Figure and table
# ---------------------------------------------------------------------------

def plot_cumulative_returns(returns_df: pd.DataFrame, save_path: str) -> None:
    """Figure 4 equivalent: cumulative wealth curves for all methods."""
    sns.set_theme(style='whitegrid')
    palette = sns.color_palette('tab10', n_colors=len(returns_df.columns))
    fig, ax = plt.subplots(figsize=(13, 7))

    for col, colour in zip(returns_df.columns, palette):
        cumret = (1.0 + returns_df[col]).cumprod()
        ax.plot(cumret.index, cumret.values, label=col,
                linewidth=1.6, color=colour)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Wealth (initial = 1)', fontsize=12)
    ax.set_title(
        'Figure 4: Min-Variance Portfolio — Cumulative OOS Returns\n'
        '(FF 48 Industries, Jan 1996 – Aug 2025)',
        fontsize=12,
    )
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%Y')
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {save_path}")


def save_metrics_table(returns_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """Table 10 equivalent: annualised metrics for all methods."""
    rows = []
    for method in ALL_METHODS:
        if method not in returns_df.columns:
            continue
        m = compute_metrics(returns_df[method])
        m['Method'] = method
        rows.append(m)

    metrics_df = (
        pd.DataFrame(rows)
        .set_index('Method')
        [['Cumulative Return', 'Ann. Return', 'Sharpe', 'Sortino']]
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_df.to_csv(save_path)
    print(f"Table saved → {save_path}")
    return metrics_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 5: Minimum-Variance Portfolio Backtest")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load / download data
    # ------------------------------------------------------------------
    print("\n[1] Loading FF48 data …")
    df = load_data().sort_index()
    df = df.loc['1981-01-01':]
    print(f"    {df.index[0].date()} – {df.index[-1].date()}, "
          f"{len(df)} months, {df.shape[1]} industries")

    # ------------------------------------------------------------------
    # 2. Date ranges
    # ------------------------------------------------------------------
    tuning_dates = df.index[
        (df.index >= TUNING_START) & (df.index <= TUNING_END)
    ]
    test_dates = df.index[
        (df.index >= TEST_START) & (df.index <= TEST_END)
    ]
    print(f"    Tuning: {len(tuning_dates)} months  |  Test: {len(test_dates)} months")

    # ------------------------------------------------------------------
    # 3. Hyperparameter tuning
    # ------------------------------------------------------------------
    print("\n[2] Tuning rho for methods with ambiguity radius …")
    method_rho = {m: 0.0 for m in METHODS_NO_RHO}
    for method in METHODS_WITH_RHO:
        print(f"    {method} …", end=' ', flush=True)
        best_rho = tune_rho(df, tuning_dates, method, RHO_GRID, WINDOW)
        method_rho[method] = best_rho
        print(f"best ρ = {best_rho:.3g}")

    # ------------------------------------------------------------------
    # 4. OOS backtest
    # ------------------------------------------------------------------
    print("\n[3] Running OOS backtest …")
    returns_df = run_backtest(df, test_dates, method_rho, WINDOW)
    print(f"    Done — {len(returns_df)} months × {len(returns_df.columns)} methods")

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    metrics_df = save_metrics_table(returns_df, TABLE_PATH)
    print("\n=== Table 10: Performance Metrics (annualised) ===")
    print(metrics_df.to_string(float_format=lambda x: f'{x:+.4f}'))

    # ------------------------------------------------------------------
    # 6. Figure
    # ------------------------------------------------------------------
    plot_cumulative_returns(returns_df, FIG_PATH)

    print("\nExperiment 5 complete.")


if __name__ == '__main__':
    main()
