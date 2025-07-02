import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance, skew, kurtosis, skewnorm, genpareto, t

np.random.seed(42)

# -------------------------
# Setup
# -------------------------
mean_returns = np.array([0.01, 0.005, -0.002])
cov_matrix = np.array([
    [0.02, 0.015, -0.005],
    [0.015, 0.03, -0.008],
    [-0.005, -0.008, 0.01]
])
stddevs = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(stddevs, stddevs)

portfolio_weights = np.array([0.6, 0.3, 0.1])
n_scenarios = 1000

factor_shocks = np.random.multivariate_normal(mean_returns, cov_matrix, n_scenarios)
portfolio_returns = factor_shocks @ portfolio_weights

alpha = 0.01
VaR = np.percentile(portfolio_returns, 100 * alpha)
ES = portfolio_returns[portfolio_returns <= VaR].mean()

print(f"VaR (99%): {VaR:.4f}, ES: {ES:.4f}")

# -------------------------
# True t-copula
# -------------------------
def generate_t_copula_samples(corr_matrix, df, n_samples):
    d = corr_matrix.shape[0]
    g = np.random.standard_normal((n_samples, d))
    chol = np.linalg.cholesky(corr_matrix)
    Z = g @ chol.T
    chi2 = np.random.chisquare(df, n_samples) / df
    T_ = Z / np.sqrt(chi2)[:, None]
    U = t.cdf(T_, df)
    return U

def t_copula_test(corr_matrix, mean_returns, cov_matrix, weights, df=5, n_scenarios=1000):
    U = generate_t_copula_samples(corr_matrix, df, n_scenarios)
    factor_shocks = np.zeros_like(U)
    for i in range(U.shape[1]):
        factor_shocks[:, i] = np.quantile(np.random.normal(mean_returns[i], np.sqrt(cov_matrix[i, i]), 10000), U[:, i])
    returns = factor_shocks @ weights

    plt.hist(returns, bins=50, alpha=0.7, color='purple', label='t-Copula')
    plt.title('Portfolio Returns from t-Copula')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# -------------------------
# Skewness stress
# -------------------------
def skewness_stress(factor_shocks, weights):
    a = -5
    skewed_samples = skewnorm.rvs(a, loc=mean_returns[0], scale=np.sqrt(cov_matrix[0,0]), size=n_scenarios)
    skewed_factors = factor_shocks.copy()
    skewed_factors[:,0] = skewed_samples
    skewed_returns = skewed_factors @ weights

    plt.hist(skewed_returns, bins=50, alpha=0.7, color='orange', label='Skewed')
    plt.hist(portfolio_returns, bins=50, alpha=0.5, label='Base')
    plt.legend()
    plt.title('Base vs Skew-Stressed Returns')
    plt.show()

# -------------------------
# Historical scenario
# -------------------------
def historical_scenario(weights):
    historical_shock = np.array([-0.4, 0.25, -0.05])
    historical_return = historical_shock @ weights
    print(f"GFC-like scenario return: {historical_return:.4f}")

# -------------------------
# Conditional ES
# -------------------------
def conditional_es(factor_shocks, returns, alpha=0.01, q=0.95):
    threshold = np.quantile(factor_shocks[:, 0], q)
    subset = returns[factor_shocks[:, 0] > threshold]
    if len(subset) > 0:
        ces = subset[subset <= np.percentile(subset, 100 * alpha)].mean()
        print(f"Conditional ES: {ces:.4f}")
    else:
        print("No samples beyond conditional threshold.")

# -------------------------
# Dynamic paths
# -------------------------
def dynamic_paths(n_steps=10, phi=0.8, sigma=0.1):
    path = [0]
    for _ in range(n_steps-1):
        path.append(phi * path[-1] + np.random.normal(0, sigma))
    plt.plot(path)
    plt.title('AR(1) Simulated Stress Path')
    plt.show()

# -------------------------
# EVT tail fit
# -------------------------
def evt_tail_fit(returns):
    losses = -returns
    threshold = np.percentile(losses, 95)
    tail_losses = losses[losses > threshold] - threshold
    if len(tail_losses) > 0:
        params = genpareto.fit(tail_losses)
        print(f"GPD shape: {params[0]:.4f}, scale: {params[2]:.4f}")
    else:
        print("No data above EVT threshold.")

# -------------------------
# Adversarial scenario
# -------------------------
def joint_adversarial_scenario(weights):
    print("Placeholder: Solve optimization problem to find worst-case joint shock vector.")

# -------------------------
# Run all advanced tests
# -------------------------
t_copula_test(corr_matrix, mean_returns, cov_matrix, portfolio_weights)
skewness_stress(factor_shocks, portfolio_weights)
historical_scenario(portfolio_weights)
conditional_es(factor_shocks, portfolio_returns)
dynamic_paths()
evt_tail_fit(portfolio_returns)
joint_adversarial_scenario(portfolio_weights)
