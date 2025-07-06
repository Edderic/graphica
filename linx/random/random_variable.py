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
    
    def __init__(self, name=None, **kwargs):
        """
        Initialize random variable.
        
        Parameters:
            name: str, optional
                Name of the random variable.
            **kwargs: dict
                Additional parameters for the specific distribution.
        """
        self.name = name
        self.parents = []
        self._process_parameters(**kwargs)
    
    def _process_parameters(self, **kwargs):
        """
        Process parameters for the random variable.
        Override in subclasses to handle specific parameters.
        """
        pass
    
    def set_parents(self, parents):
        """
        Set parent random variables.
        
        Parameters:
            parents: list[RandomVariable]
                List of parent random variables.
        """
        self.parents = parents
    
    def get_parents(self):
        """
        Get parent random variables.
        
        Returns:
            list[RandomVariable]
                List of parent random variables.
        """
        return self.parents
    
    @abstractmethod
    def pdf(self, x, **kwargs):
        """
        Probability density function.
        
        Parameters:
            x: array-like
                Points at which to evaluate the probability density function.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).
                
        Returns:
            array-like: Probability density values at the given points.
        """
        pass
    
    @abstractmethod
    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.
        
        Parameters:
            x: array-like
                Points at which to evaluate the log probability density function.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).
                
        Returns:
            array-like: Log probability density values at the given points.
        """
        pass
    
    @abstractmethod
    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the distribution.
        
        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).
                
        Returns:
            array-like: Random samples from the distribution.
        """
        pass
    
    def __repr__(self):
        """String representation of the random variable."""
        if self.name:
            return f"{self.__class__.__name__}(name='{self.name}')"
        else:
            return f"{self.__class__.__name__}()"
    
    def __str__(self):
        """String representation of the random variable."""
        return self.__repr__() 