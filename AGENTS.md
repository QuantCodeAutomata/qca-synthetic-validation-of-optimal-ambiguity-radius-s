# AGENTS.md — Project Knowledge Base

## Project Overview
SCOPE (Spectral Covariance with Optimal Perturbation Estimator) research project.
Implements covariance estimation under Wasserstein / KL / Symmetrised-Stein
divergence balls and benchmarks against LW-L, LW-NL, W-DRCOE, WISE.

## Repository Structure
```
src/
  scope.py              — SCOPE estimator, divergences, phi maps, W-DRCOE, WISE
  benchmark_estimators.py — sample_cov, lw_linear_shrinkage, lw_nonlinear_shrinkage,
                            unified estimate_cov()
exp/
  exp_5_portfolio.py    — FF48 min-variance portfolio backtest (Jan 1996–Aug 2025)
data/
  ff48_returns.csv      — cached Fama-French 48-industry monthly VW returns (decimal)
results/
  exp5_portfolio_cumulative_returns.png
  exp5_portfolio_metrics_table.csv
tests/
  __init__.py
  test_scope.py         — 15 unit tests for src/scope.py
  test_experiments.py   — 3 smoke tests (exp1, exp2, exp4 mini inline versions)
```

## Key Design Decisions

### SCOPE estimator (src/scope.py)
- Eigenvectors preserved from Sigma_hat; only eigenvalues shrunk toward 1/sqrt(tau)
- tau (plug-in) = p / ||Sigma_hat||_F^2
- rho >= rho_max → scalar-target matrix (1/sqrt(tau) * I)
- phi is strictly monotone increasing in b (for gamma > 0) → eigenvector order preserved
  ONLY when rho < rho_max; at rho >= rho_max all eigenvalues equal → eigenvectors undefined

### Financial data conventions (exp_5_portfolio.py)
- Centered, Bessel-corrected covariance (zero_mean=False, n-1 denominator)
- tau computed from Sigma_hat (centered), not from raw data
- LW-L / LW-NL: assume_centered=False
- Rolling window = 60 months; for month t use [t-60, t-1] months, observe return at t

### test_scope.py — Key gotchas
- test_scope_eigenvector_preservation: MUST use rho < rho_max (use rho = rho_max * 0.1)
  At rho >= rho_max, SCOPE returns scalar-target (all equal eigenvalues → arbitrary eigvecs)
- test_exp2_hd_comparison_runs: n=8 < p=10 → sample cov is rank-deficient (PSD not PD)
  Only regularised estimators (LW, SCOPE, W-DRCOE, WISE) asserted PD for n < p

## Experiment 1 — Radius Scaling (exp_1_radius_scaling.py)

### Results (seed=42, p=5, eigenvalues=[1,2,3,4,5], 141 n-values × 50 reps)
| Divergence | alpha   | beta   | R²     |
|------------|---------|--------|--------|
| KL         | −2.0016 | 5.0665 | 0.9881 |
| W          | −1.7936 | 4.7287 | 0.9870 |
| SS         | −1.8879 | 5.2807 | 0.9887 |

All structural checks PASS: positive definite, eigenvectors preserved (overlap=1.0),
scalar limit (L∞ err=0), Sigma_hat limit (L∞ err < 1e-4).

### Critical algorithmic speedup: gamma sweep (not rho sweep)
The rho-grid search would need O(N_RHO × outer_brentq × p × inner_brentq) evaluations
per (n, rep) pair — prohibitively slow. Instead exploit the bijection:

    gamma  ↔  rho = Σ_i d(phi(tau, gamma, lambda_i), lambda_i)

Sweep 100 log-spaced gamma values per (n, rep, div), compute (rho, loss) pairs from the
closed-form phi functions (kl_phi_v / wasserstein_phi_v / stein_phi_v), then interpolate
onto the canonical 300-point rho-grid. Gives identical rho_tilde_n at ~150x less cost.

### Runtime
- 141 n-values × 50 reps × 3 divergences, 100 gamma pts, 4 workers: ~11 min
- stdout output is block-buffered when piped to file — progress lines appear only on exit
- Other experiments running simultaneously (exp_2, exp_4) increase wall time 2-3x

## Running Tests
```bash
cd /workspace/project
python -m pytest tests/ -v
```
All 18 tests pass in ~10 seconds (no network calls needed).

## Running Experiment 5
```bash
cd /workspace/project
python exp/exp_5_portfolio.py
```
Downloads FF48 data (~60 months download), caches to data/ff48_returns.csv.
Full run takes ~15-30 minutes (tuning 5 methods × 50 rho values × 120 months).
Saves figure to results/ and metrics CSV to results/.

## Experiment 2 — High-Dimensional Covariance Estimation (exp_2_hd_comparison.py)

Benchmarks W-SCOPE vs sample, LW-L, LW-NL, W-DRCOE, WISE across p∈{150,200} and
multiple sample sizes n, using 50 Monte Carlo repetitions per (p, n).

### DGP
- Sigma_0 = V^T diag(1..p) V via make_true_covariance(p, eigenvalues=np.arange(1,p+1), seed=42)
- p=150: n ∈ {50, 70, 100, 120, 150, 200, 300}
- p=200: n ∈ {50, 100, 120, 150, 170, 200, 300, 400}
- seed = p*10000 + n*100 + rep  (per-rep RNG via np.random.default_rng)

### Oracle Rho Tuning
- 100 log-spaced rho values in [1e-5, 2e3]
- For each (p,n): run all 50 reps × 100 rho values, average the target loss, pick best rho
- W-SCOPE: oracle_combined_loss  → consistently selects rho_max≈2000 (full shrinkage)
- W-DRCOE: oracle_frobenius_loss → consistently selects rho=1e-5 (≈sample cov)
- WISE:    oracle_stein_loss     → selects moderate rho that decreases as n grows

### Loss Evaluation
- Combined Stein-Frobenius loss: tau_star = p / ||Sigma_0||_F^2
  - NaN for sample cov when n ≤ p (singular, cannot invert)
  - W-SCOPE uses tau_hat = p / ||Sigma_hat||_F^2 internally
- Relative eigenvalue error: relative_eigenvalue_error(Sigma_est, Sigma_0) from scope.py

### Key Results
- W-SCOPE wins on combined loss for n ≥ p (e.g., p=150, n=200: 727.7 vs LW-L 728.4)
- LW-L most robust for n < p (combined loss ~730-1042 regardless of n/p)
- LW-NL catastrophically fails for n ≤ p (loss ~1e15-1e16; kernel bandwidth → 0 with
  many clipped-zero eigenvalues; not a code bug — genuine limitation of estimator)
- WISE fails for small n < p (loss ~1e21; rho=1e-5 selected but estimate is degenerate)
- W-DRCOE fails for n < p (loss ~1e9-2e11; rho=1e-5 ≈ sample cov)
- On eigenvalue error: sample cov best, W-DRCOE ≈ sample, W-SCOPE intermediate, LW-L worst

### Computational Profile
- Tuning phase: 50 reps × 100 rho values per (p,n) → bottleneck is n < p configs
  - p=200, n=50:  ~180s tuning (scope_estimate invokes nested brentq per eigenvalue)
  - p=200, n=120: ~130s tuning
  - p=200, n=170: ~38s tuning (fewer zero eigenvalues → smaller rho_max → faster brentq)
  - p=200, n=300: ~39s tuning (n > p, full rank sample cov)
- Evaluation phase: 50 reps per method × 6 methods → ~35-140s depending on n
- Total wall time: ~4–5 hours (dominated by p=200, n=50,100 tuning blocks)

### Outputs
- results/exp2_combined_loss_table.csv   — 6 methods × 15 (p,n) columns
- results/exp2_eigenvalue_error_table.csv

---

## Experiment 3 — Hyperspectral Anomaly Detection (exp_3_hyperspectral.py)

Simulates 5 hyperspectral datasets (real data unavailable without licenses).

### Data Generation
- Datasets: p ∈ {50,100,150,200,224}, N = p+10, seed = 42+i (i=1..5)
- Sigma_bg = Q diag(1..p) Q^T  (random orthogonal Q via sign-normalised QR)
- Anomaly mean: mu_anomaly = 3 * e_1 (anomaly in first band only); 5% anomaly pixels

### Covariance Estimation
- All 8 methods: sample, LW-L, LW-NL, W-SCOPE, KL-SCOPE, SS-SCOPE, W-DRCOE, WISE
- Sample covariance: Bessel-corrected (n-1), centered; pinv when near-singular
- rho tuning for SCOPE/DRCOE/WISE: grid logspace(-5, log10(200), 30 pts)
  Cross-validated trace proxy: minimise tr(Sigma_est(rho)^{-1} @ Sigma_hat_val)
  where val = second 50% of pixels; rho_star applied to full-data estimate

### Performance Notes
- KL-SCOPE often selects rho ≈ 1e-5 (≈ sample cov) for N > p because
  tr(Sigma_est^{-1} @ Sigma_hat) is nearly monotone-increasing in rho
- scope_estimate raises divide-by-zero in kl_div_v when half-data cov is rank-deficient;
  benign warning — eigenvalues are floored at 1e-12 before phi computation

### RX Detector
- score_i = (x_i - mu_hat)^T Sigma_est^{-1} (x_i - mu_hat)
- AUC from roc_auc_score(labels, scores)

### Outputs
- results/exp3_hyperspectral_auc_table.csv  — 8×5 AUC table (methods × datasets)
- results/exp3_auc_comparison.png           — AUC vs p line plot

---

## Experiment 4 — A/B Test Metric Learning (exp_4_ab_testing.py)

Learns a linear test direction w from training pairs and evaluates it on test pairs
via a z-statistic.

### DGP
- V = sign-normalised QR(G) with G ~ N(0,1)^{50×50}, seed=42
- Sigma_1 = V^T diag(1..50)/50 V  (control covariance)
- Sigma_2 = V^T diag(2..51)/50 V  (treatment covariance)
- mu = 0.3 * ones(50)/sqrt(50)     (treatment effect; ||mu|| = 0.3)
- A/B pair: A ~ N(0, Sigma_1), B ~ N(mu, Sigma_2)
- A/A pair: both ~ N(0, Sigma_1)
- seed = rep * 10_000 + n

### Weight Aggregation
- w_k = pinv(f(Sigma_Ak) + f(Sigma_Bk)) @ (mu_hat_Bk - mu_hat_Ak)
- w_k /= ||w_k|| + 1e-10
- w_star = mean over k of w_k

### Test z-score
- z = w_star^T (mu_hat_B - mu_hat_A) / sqrt(w^T S_A w/n + w^T S_B w/n)
  where S_A, S_B are sample covariances of the test groups (method-agnostic)

### CRITICAL Performance Issue (RESOLVED)
`scope_estimate` for W-SCOPE / SS-SCOPE calls brentq for every eigenvalue inside
the outer gamma_star brentq loop — this creates ~p × 50 nested brentq calls per
estimate call. For p=50 and 60,000 SCOPE calls per experiment, this is **hours**.

**Fix in exp_4**: Use `@numba.njit(cache=True)` Newton solvers for W and SS phi.
The outer gamma_star brentq still uses brentq (1 call), but each evaluation uses
O(p × 50-iter) machine-code Newton. Speed: ~0.5ms/call (vs ~47ms with pure Python).

Always call `_w_phi_v(1.0, 1.0, dummy)` and `_ss_phi_v(1.0, 1.0, dummy)` before the
main loop to trigger JIT compilation (avoids cold-start latency inside timing).

### Runtime (with numba)
- 20 reps × 5 n-values × 100 training pairs × 2 groups × 3 SCOPE methods
- ~250 seconds per n-value, ~22 minutes total
- Eigendecomposition (0.5ms × 60,000 = 30s) dominates after numba speedup

### Outputs
- results/exp4_ab_testing_auc_table.csv  — 7×5 mean AUC table (methods × n_train)
- results/exp4_auc_vs_n.png              — mean AUC ± std vs n line plot

---

## Package Requirements
numpy, scipy, pandas, cvxpy (OSQP solver), matplotlib, seaborn, sklearn, requests,
numba (≥0.60, used in exp_4 for JIT-compiled Newton phi loops)
