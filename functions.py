from typing import TypeVar
from abc import ABC, abstractmethod
import numpy as np
from numpy.polynomial.polynomial import Polynomial


__all__ = ['RealFunction', 'CstFunction', 'AffineFunction', 'Interval']

T = TypeVar('T', float, np.ndarray)

Interval = tuple[float, float]

class RealFunction(ABC):
    @abstractmethod
    def __call__(self, x: T) -> T:
        NotImplemented
    
    def __add__(self, x) -> "RealFunction":
        return x + self
    
    def __sub__(self, x) -> "RealFunction":
        return self + -x
    
    @abstractmethod
    def __neg__(self) -> "RealFunction":
        NotImplemented

    @abstractmethod
    def integral(self, interval: Interval) -> float:
        NotImplemented
    
    @abstractmethod
    def square(self) -> "RealFunction":
        NotImplemented

class CstFunction(RealFunction):
    
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, x: T) -> T:
        return 0.0 * x + self.value
    
    def __add__(self, x) -> "RealFunction":
        if isinstance(x, (float, int)):
            return CstFunction(self.value + x)
        elif isinstance(x, CstFunction):
            return CstFunction(self.value + x.value)
        else:
            return super().__add__(x)
    
    def __neg__(self) -> "RealFunction":
        return CstFunction(- self.value)

    def integral(self, interval: Interval) -> float:
        return (interval[1] - interval[0]) * self.value
    
    def square(self) -> RealFunction:
        return CstFunction(self.value ** 2)

class AffineFunction(RealFunction):
    
    def __init__(self, slope: float, intercept: float):
        super().__init__()
        self.slope = slope
        self.intercept = intercept
    
    def __call__(self, x: T) -> T:
        if isinstance(x, (float, int)):
            return self.slope * x + self.intercept
    
    def __add__(self, x) -> RealFunction:
        if isinstance(x, (float, int)):
            return AffineFunction(self.slope, self.intercept + x)
        elif isinstance(x, CstFunction):
            return self + x.value
        elif isinstance(x, AffineFunction):
            return AffineFunction(self.slope + x.slope, self.intercept + x.intercept)
        else:
            return super().__add__(x)
    
    def __neg__(self) -> RealFunction:
        return AffineFunction(-self.slope, -self.intercept)
    
    def integral(self, interval: Interval) -> float:
        return (self.slope * (interval[1] + interval[0]) / 2 + self.intercept) * (interval[1] - interval[0])
    
    def square(self) -> RealFunction:
        return PolynomialFunction([self.intercept**2, 2*self.slope*self.intercept, self.slope**2])
        

class PolynomialFunction(RealFunction):
    
    def __init__(self, *args, **kwargs):
        self.poly = Polynomial(*args, **kwargs)
    
    def __call__(self, x: T) -> T:
        return self.poly(x)
    
    def __add__(self, x) -> RealFunction:
        return Polynomial(self.poly + x)
    
    def __neg__(self) -> RealFunction:
        return Polynomial(-self.poly)
    
    def integral(self, interval: Interval) -> float:
        integ = self.poly.integ()
        return integ(interval[1]) - integ(interval[0])
    
    def square(self) -> RealFunction:
        return self.poly ** 2

