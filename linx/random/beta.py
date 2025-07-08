"""
Beta random variable class
"""
import numpy as np
from scipy.stats import beta
from .random_variable import RandomVariable


class Beta(RandomVariable):
    """
    Beta random variable.
    
    Parameters:
        name: str, optional
            Name of the random variable.
        alpha: float
            First shape parameter (alpha > 0).
        beta_param: float
            Second shape parameter (beta > 0).
    """
    
    def __init__(self, name=None, alpha=1.0, beta_param=1.0, **kwargs):
        """
        Initialize Beta random variable.
        
        Parameters:
            name: str, optional
                Name of the random variable.
            alpha: float
                First shape parameter (alpha > 0).
            beta_param: float
                Second shape parameter (beta > 0).
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta_param = beta_param
        
        # Validate parameters
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta_param <= 0:
            raise ValueError("beta must be positive")
    
    def _process_parameters(self, **kwargs):
        """Process parameters for the beta distribution."""
        # Parameters are handled in __init__
        pass
    
    def pdf(self, x, **kwargs):
        """
        Probability density function for beta distribution.
        
        Parameters:
            x: float or array-like
                Points in [0, 1] at which to evaluate the PDF.
            **kwargs: dict
                Additional parameters (ignored for beta).
                
        Returns:
            float or array-like: Probability density at the given point(s).
        """
        return beta.pdf(x, self.alpha, self.beta_param)
    
    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.
        
        Parameters:
            x: float or array-like
                Points in [0, 1] at which to evaluate the log PDF.
            **kwargs: dict
                Additional parameters (ignored for beta).
                
        Returns:
            float or array-like: Log probability density at the given point(s).
        """
        return beta.logpdf(x, self.alpha, self.beta_param)
    
    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the beta distribution.
        
        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Additional parameters (ignored for beta).
                
        Returns:
            float or array-like: Random samples from the distribution.
        """
        return beta.rvs(self.alpha, self.beta_param, size=size)
    
    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function.
        
        Parameters:
            x: float or array-like
                Points in [0, 1] at which to evaluate the CDF.
            **kwargs: dict
                Additional parameters (ignored for beta).
                
        Returns:
            float or array-like: Cumulative probability at the given point(s).
        """
        return beta.cdf(x, self.alpha, self.beta_param)
    
    def __repr__(self):
        """String representation of the beta random variable."""
        if self.name:
            return f"Beta(name='{self.name}', alpha={self.alpha}, beta={self.beta_param})"
        else:
            return f"Beta(alpha={self.alpha}, beta={self.beta_param})"
    
    def __str__(self):
        """String representation of the beta random variable."""
        return self.__repr__() 