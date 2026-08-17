# SCOPE Estimator Experiments

Synthetic and real-data validation of the **SCOPE** (Spectral Covariance with Optimal Perturbation Estimator) family for joint covariance-precision estimation.

## Experiments

| ID     | Title                                                        |
|--------|--------------------------------------------------------------|
| exp_1  | Synthetic validation of optimal ambiguity radius scaling     |
| exp_2  | Synthetic high-dimensional covariance estimation comparison  |
| exp_3  | Hyperspectral anomaly detection with SCOPE-based covariance  |
| exp_4  | Synthetic A/B test metric learning with covariance regularisation |
| exp_5  | Minimum-variance portfolio backtest with SCOPE estimators    |

## Repository Structure

```
├── data/               # Data loading scripts / downloaded datasets
├── src/
│   ├── scope.py        # Core SCOPE estimator (KL, W, SS divergences)
│   └── benchmark_estimators.py  # LW-L, LW-NL, W-DRCOE, WISE
├── exp/
│   ├── exp_1_radius_scaling.py
│   ├── exp_2_hd_comparison.py
│   ├── exp_3_hyperspectral.py
│   ├── exp_4_ab_testing.py
│   └── exp_5_portfolio.py
├── tests/
│   ├── test_scope.py
│   └── test_experiments.py
└── results/
    └── RESULTS.md
```

## Setup

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas cvxpy
```

## Running Experiments

```bash
# Run all experiments
python exp/exp_1_radius_scaling.py
python exp/exp_2_hd_comparison.py
python exp/exp_3_hyperspectral.py
python exp/exp_4_ab_testing.py
python exp/exp_5_portfolio.py

# Run tests
pytest tests/ -v
```

## Key Implementation Notes

- **SCOPE** estimator uses eigenvalue shrinkage (no SDP required).
- **Eigenvectors** from `Sigma_hat` are always preserved.
- **Scalar target**: `sqrt(1/tau) * I` where `tau = p / ||Sigma_hat||_F^2`.
- **Dual variable** `gamma*` solved via `scipy.optimize.brentq`.
- **LW-NL** uses Ledoit & Wolf (2012) analytical nonlinear shrinkage formula.
- **W-DRCOE** and **WISE** derived from KKT of respective Wasserstein DRO problems.
