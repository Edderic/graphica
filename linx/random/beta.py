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
        a: float
            First shape parameter (a > 0).
        b: float
            Second shape parameter (beta > 0).
    """

    def __init__(self, name=None, a=1.0, b=1.0, **kwargs):
        """
        Initialize Beta random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            a: float
                First shape parameter (a > 0).
            b: float
                Second shape parameter (beta > 0).
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.a = a
        self.b = b

        # Validate parameters
        if a <= 0:
            raise ValueError("a must be positive")
        if b <= 0:
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
        return beta.pdf(x, self.a, self.b)

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

        new_kwargs = {
            'a': self.a,
            'b': self.b
        }

        new_kwargs.update(kwargs)

        return beta.logpdf(x, **new_kwargs)

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
        return beta.rvs(self.a, self.b, size=size)

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
        return beta.cdf(x, self.a, self.b)

    def __repr__(self):
        """String representation of the beta random variable."""
        if self.name:
            return f"Beta(name='{self.name}', a={self.a}, beta={self.b})"
        else:
            return f"Beta(a={self.a}, beta={self.b})"

    def __str__(self):
        """String representation of the beta random variable."""
        return self.__repr__()
