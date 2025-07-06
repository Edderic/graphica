"""
Uniform distribution implementation.
"""
import numpy as np
from .random_variable import RandomVariable


class Uniform(RandomVariable):
    """
    Uniform distribution.
    
    Parameters:
        low: float
            Lower bound of the uniform distribution.
        high: float
            Upper bound of the uniform distribution.
    """
    
    def _process_parameters(self, low=0.0, high=1.0, **kwargs):
        """
        Process parameters for uniform distribution.
        
        Parameters:
            low: float
                Lower bound of the distribution.
            high: float
                Upper bound of the distribution.
        """
        if low >= high:
            raise ValueError("Lower bound must be less than upper bound")
        
        self.low = low
        self.high = high
        self.width = high - low
    
    def pdf(self, x, **kwargs):
        """
        Probability density function of the uniform distribution.
        
        f(x) = 1/(high-low) for low <= x <= high, 0 otherwise.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= self.low) & (x <= self.high)
        result[mask] = 1.0 / self.width
        return result
    
    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.
        
        log(f(x)) = -log(high-low) for low <= x <= high, -inf otherwise.
        """
        x = np.asarray(x)
        result = np.full_like(x, -np.inf, dtype=float)
        mask = (x >= self.low) & (x <= self.high)
        result[mask] = -np.log(self.width)
        return result
    
    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the uniform distribution.
        """
        return np.random.uniform(self.low, self.high, size=size)
    
    def cdf(self, x):
        """
        Cumulative distribution function of the uniform distribution.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= self.low) & (x <= self.high)
        result[mask] = (x[mask] - self.low) / self.width
        result[x > self.high] = 1.0
        return result
    
    def __repr__(self):
        """String representation of the uniform distribution."""
        if self.name:
            return f"Uniform(name='{self.name}', low={self.low}, high={self.high})"
        else:
            return f"Uniform(low={self.low}, high={self.high})"
    
    def __str__(self):
        """String representation of the uniform distribution."""
        return self.__repr__() 