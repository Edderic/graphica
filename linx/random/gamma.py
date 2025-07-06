"""
Gamma distribution implementation.
"""
import numpy as np
from .random_variable import RandomVariable

class Gamma(RandomVariable):
    """
    Gamma distribution.
    
    Parameters:
        shape: float
            Shape parameter (k or alpha), must be > 0.
        scale: float, default=1.0
            Scale parameter (theta), must be > 0.
    """
    def __init__(self, shape, scale=1.0):
        if shape <= 0:
            raise ValueError("Shape parameter must be positive")
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")
        self.shape = shape
        self.scale = scale
    
    def pdf(self, x):
        """
        Probability density function of the gamma distribution.
        """
        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.pdf(x, a=self.shape, scale=self.scale)
    
    def logpdf(self, x):
        """
        Logarithm of the probability density function.
        """
        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.logpdf(x, a=self.shape, scale=self.scale)
    
    def sample(self, size=None):
        """
        Generate random samples from the gamma distribution.
        """
        return np.random.gamma(shape=self.shape, scale=self.scale, size=size)
    
    def cdf(self, x):
        """
        Cumulative distribution function of the gamma distribution.
        """
        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.cdf(x, a=self.shape, scale=self.scale)
    
    def __repr__(self):
        return f"Gamma(shape={self.shape}, scale={self.scale})"
    
    def __str__(self):
        return self.__repr__() 