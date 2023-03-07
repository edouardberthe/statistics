from dataclasses import dataclass

__all__ = ['Market']

@dataclass
class Market:
    S0: float
    r: float
    vol: float
