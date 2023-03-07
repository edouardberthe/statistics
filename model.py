from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from functions import *

__all__ = ['EquityModel', 'BlackScholesModel', 'RateModel', 'HoLeeModel', 'DeterministicRateModel', 'Model']


class EquityModel(ABC):
    @abstractmethod
    def get_volatility(self, t1, t2, S):
        NotImplemented
    
    def __repr__(self) -> str:
        return str(self)

class BlackScholesModel(EquityModel):
    def __init__(self, sigma):
        self.sigma = sigma
    def get_volatility(self, interval: Interval, S) -> RealFunction:
        return CstFunction(self.sigma)
    def __str__(self) -> str:
        return f"BS Model ({self.sigma:.1%})"

class RateModel(ABC):
    @abstractmethod
    def get_volatility(self, interval: Interval, T: float) -> RealFunction:
        """Volatility of the rate abstract factor"""
        NotImplemented
    
    def __str__(self) -> str:
        return "Deterministic"
    
    def __repr__(self) -> str:
        return str(self)

class DeterministicRateModel(RateModel):
    def get_volatility(self, interval: Interval, T: float) -> RealFunction:
        return CstFunction(0.0)
    
class HoLeeModel(RateModel):
    def __init__(self, vol: float):
        super().__init__()
        self.vol = vol
    def get_volatility(self, interval: Interval, T: float) -> RealFunction:
        return AffineFunction( -self.vol, T * self.vol) 
    def __str__(self) -> str:
        return f"Ho-Lee Model ({self.vol:.1%})"
    def __repr__(self) -> str:
        return str(self)

@dataclass
class Model:
    equity_model: EquityModel
    rate_model: RateModel = field(default_factory=DeterministicRateModel)
    
    def __str__(self) -> str:
        return f"{self.equity_model}, {self.rate_model}"
