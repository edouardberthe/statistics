from scipy import stats
import numpy as np

__all__ = ['bs_price']

def bs_price(S0, K, r, T, sigma):
    F0 = S0 * np.exp(r * T)
    x = np.log(S0/K) + r * T
    s = sigma * np.sqrt(T)
    d1 = x / s + s/2
    d2 = d1 - s
    return np.exp(-r*T) * (F0 * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))