from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from scipy import stats

from model import *
from payoff import *
from market import Market

__all__ = ['McResult', 'mc_price']


@dataclass
class McResult:
    prices: list[float]
    
    def mean(self):
        return np.mean(self.prices)
    
    def stderr(self, alpha: float = 0.05) -> float:
        p = stats.norm.ppf(1 - alpha/2)
        return np.std(self.prices) * p / np.sqrt(len(self.prices))
    
    def CI(self, alpha: float = 0.05) -> tuple[float,float]:
        stderr = self.stderr(alpha)
        mean = self.mean()
        return [mean - stderr, mean + stderr]
    
    def __str__(self) -> str:
        return f"{self.mean():.5f} +/- {self.stderr():.5f}"
    
    def __repr__(self) -> str:
        return str(self)


Scheme = list[float]

def mc_price(payoff: Callable, model: Model, market: Market, paths: int, scheme: Scheme, seed: int = 3):
    if isinstance(model.rate_model, DeterministicRateModel):
        return mc_price_det_rate(payoff, model, market, paths, scheme, seed)
    else:
        return mc_price_sto_rate(payoff, model, market, paths, scheme, seed)

def mc_price_det_rate(payoff: Callable, model: Model, market: Market, paths: int, scheme: Scheme, seed):
    S0, r = market.S0, market.r
    equity_model = model.equity_model
    if seed is not False:
        np.random.seed(seed)
    logS = np.log(S0)
    for i in range(len(scheme) - 1):
        t1, t2 = scheme[i], scheme[i+1]
        dt = t2 - t1
        dW = np.random.randn(paths) * np.sqrt(dt)
        vol = equity_model.get_volatility((t1, t2), np.exp(logS))
        eq_vol = np.sqrt(vol.square().integral((t1, t2))/dt)
        logS += (r - eq_vol * eq_vol / 2) * dt + eq_vol * dW
    S = np.exp(logS)
    P = payoff(S)
    T = scheme[-1]
    return McResult(np.exp(-r*T) * P)

def mc_price_sto_rate(payoff: Callable, model: Model, market: Market, paths: int, scheme: Scheme, seed):
    S0, r = market.S0, market.r
    equity_model, rate_model = model.equity_model, model.rate_model
    if seed is not False:
        np.random.seed(seed)
    T = scheme[-1]
    logS = np.log(S0)
    logZC = np.zeros(paths) - r * T
    for i in range(len(scheme) - 1):
        t1, t2 = scheme[i], scheme[i+1]
        dt = t2 - t1
        
        S = np.exp(logS)
        equity_vol = equity_model.get_volatility((t1, t2), S)
        zc_vol = rate_model.get_volatility((t1, t2), T)
        
        step_S_var  = equity_vol.square().integral((t1, t2))
        step_zc_var = zc_vol.square().integral((t1, t2))
        
        step_S_vol  = np.sqrt(step_S_var / dt)
        step_zc_vol = np.sqrt(step_zc_var / dt)
        
        dW = np.random.randn(paths, 2) * np.sqrt(dt)
        logS  += (r -  step_S_var / 2) * dt + step_S_vol * dW[:,0]
        logZC += (r + step_zc_var / 2) * dt + step_zc_vol * dW[:,1]
    S = np.exp(logS)
    ZC = np.exp(logZC)
    P = payoff(S)
    return McResult(np.exp(-r*T) * P)
