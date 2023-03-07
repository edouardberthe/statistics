import numpy as np

__all__ = ['CallPayoff']


class CallPayoff:
    def __init__(self, K):
        self.K = K
    def __call__(self, S):
        return np.maximum(S - self.K, 0.0)