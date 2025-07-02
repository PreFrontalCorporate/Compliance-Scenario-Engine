# Compliance Scenario Engine

## Overview

The **Compliance Scenario Engine** is an advanced, academically rigorous stress testing suite designed to simulate extreme market and macroeconomic scenarios for financial ML models and portfolios. It supports regulatory requirements (e.g., CCAR, Basel III) and deep academic analysis.

---

## Features

✅ Simulate multivariate factor shocks with true t-copula modeling to capture heavy tail dependence.
✅ Explicit skewness stressing using skew-normal perturbations.
✅ Historical crisis scenario backtests (e.g., 2008 GFC).
✅ Conditional Expected Shortfall (CoES) calculations.
✅ Dynamic autoregressive stress paths.
✅ Extreme Value Theory (EVT) tail fitting for ultra-extreme events.
✅ Placeholder for joint adversarial scenario optimization.

---

## Risk Metrics

* **Value-at-Risk (VaR)**: 99% quantile of simulated loss distribution.
* **Expected Shortfall (ES)**: Mean loss beyond VaR.
* **Conditional ES**: Expected shortfall under conditional factor extremes.
* **Tail Shape (EVT)**: Estimated using Generalized Pareto Distribution.

---

## How it works

### Core Setup

* Defines economic factor means and covariance matrix.
* Generates baseline shocks and portfolio returns.
* Prints base VaR and ES.

### Advanced Modules

#### ✅ True t-Copula Test

Simulates heavy-tail dependence shocks using a manual t-copula implementation with explicit Cholesky decomposition.

#### ✅ Skewness Stress

Perturbs first factor (e.g., equity) using skew-normal distribution to simulate left-skewed shocks.

#### ✅ Historical Scenario

Applies a deterministic shock vector resembling real-world crises.

#### ✅ Conditional ES

Calculates ES conditional on extreme factor thresholds (e.g., factor > 95th percentile).

#### ✅ Dynamic AR(1) Stress Path

Generates sequential stress evolution paths to study path-dependent vulnerabilities.

#### ✅ EVT Tail Fit

Fits Generalized Pareto Distribution to portfolio loss tail to extrapolate rare event risks.

#### ✅ Adversarial Scenario Placeholder

Outlines structure for future joint worst-case optimization studies.

---

## Example Output

* **VaR (99%)**: \~-0.27
* **ES**: \~-0.31
* **GFC-like scenario return**: \~-0.17
* **Conditional ES**: \~0.12
* **EVT shape parameter**: \~0.01

![t-Copula Histogram](./t-copula-histogram.png)
![Skew Stress Histogram](./skew-stress-histogram.png)
![AR(1) Stress Path](./ar1-path.png)

---

## Usage

```bash
python compliance-scenario-engine.py
```

---

## Requirements

* Python 3.8+
* numpy
* scipy
* matplotlib

---

## References

* McNeil, A.J., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*.
* Sklar, A. (1959). *Copula functions*.
* Basel Committee on Banking Supervision. (2009). *Principles for sound stress testing practices*.

---

## Reproducibility

All figures and metrics can be exactly reproduced using the included code, ensuring compliance with academic and regulatory standards.

---

## License

MIT License
