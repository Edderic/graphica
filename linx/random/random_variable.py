"""
RandomVariable abstract class for probability distributions.
"""
from abc import ABC, abstractmethod
import numpy as np


class RandomVariable(ABC):
    """
    Abstract base class for random variables.
    
    This class defines the interface that all random variables must implement.
    Random variables can have probability density functions (pdf), log probability
    density functions (logpdf), and sampling capabilities.
    """
    
    @abstractmethod
    def pdf(self, x):
        """
        Probability density function.
        
        Parameters:
            x: array-like
                Points at which to evaluate the probability density function.
                
        Returns:
            array-like: Probability density values at the given points.
        """
        pass
    
    @abstractmethod
    def logpdf(self, x):
        """
        Logarithm of the probability density function.
        
        Parameters:
            x: array-like
                Points at which to evaluate the log probability density function.
                
        Returns:
            array-like: Log probability density values at the given points.
        """
        pass
    
    @abstractmethod
    def sample(self, size=None):
        """
        Generate random samples from the distribution.
        
        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
                
        Returns:
            array-like: Random samples from the distribution.
        """
        pass
    
    def __repr__(self):
        """String representation of the random variable."""
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        """String representation of the random variable."""
        return self.__repr__() 