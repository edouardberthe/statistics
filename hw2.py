import numpy as np
from scipy import stats

# Constants
bps = np.sqrt(252 / 10000)

# Product
K = 90
T = 1

# Market

# Stock
S0 = 100
sigma = 0.15

# Rates
r0 = 0.005
sigma_r = 5 * bps
lambda_r = 0.20

# Correlations
rho_sr = 0.10  # Equity/r
# rho = -0.10              # Intra HW2

# Pricing Configuration
N = 100                 # Time steps in Euler scheme
M = 1000                # Monte-Carlo paths
dt = T / N

# We have under risk neutral measure:
# dS(t)/S(t) = r(t) dt + sigma(t) dW^S(t)
# r(t) = phi(0,t) + Y1(t) + Y2(t)
# dYi(t) = - lambda_i Yi(t) dt + sigma^i(t) dW^i(t)

# We simulate the log-forward stock and the log ZC
# X(t) = log(S(t) / ZC(t,T))
# dX(t) = -sigma^(t) / 2 dt + sigma(t) dW^S(t)
# dZC(t) / ZC(t) = r(t) dt + sigma

S = S0 * np.ones(M)
r = r0 * np.ones(M)

# C = np.array([
#     [   1, rho1, rho2],
#     [rho1,    1,  rho],
#     [rho2,  rho,    1],
# ])
# chol = np.linalg.cholesky(C)
chol = np.array([
    [1, rho_sr],
    [0, np.sqrt(1 - rho_sr * rho_sr)]])
sigma2 = sigma * sigma

for t in np.linspace(0, T, N + 1):
    dW = np.sqrt(dt) * np.random.randn(M, 2).dot(chol)
    S += S * (r * dt + sigma * dW[:, 0])
    r += - lambda_r * r * dt + sigma_r * dW[:, 1]


def payoff(S, K):
    res = S - K
    res[res < 0] = 0
    return res


def black(S, K, r, t, T, sigma):
    tau = T - t
    discount = np.exp(- r * tau)
    F = S / discount
    x = np.log(F / K)
    v = np.sqrt(tau) * sigma
    d1 = x / v + v / 2
    d2 = x / v - v / 2
    return discount * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))

prices = payoff(S, K)
print(prices.mean(), prices.std() / np.sqrt(N))
print(black(S0, K, r0, 0, T, sigma))

