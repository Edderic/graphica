"""
Binomial random variable class
"""
import numpy as np
from scipy.stats import binom
from .random_variable import RandomVariable


class Binomial(RandomVariable):
    """
    Binomial random variable.

    Parameters:
        name: str, optional
            Name of the random variable.
        n: int
            Number of trials.
        p: float
            Probability of success on each trial (0 <= p <= 1).
    """

    def __init__(self, name=None, n=1, p=0.5, **kwargs):
        """
        Initialize Binomial random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            n: int
                Number of trials.
            p: float
                Probability of success on each trial (0 <= p <= 1).
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.n = n
        self.p = p

        # Validate parameters
        if not isinstance(n, RandomVariable) and (not isinstance(n, int) or n < 0):
            raise ValueError("n must be a non-negative integer")
        if not isinstance(p, RandomVariable) and not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        if isinstance(p, RandomVariable):
            self.parents['p'] = p
        if isinstance(n, RandomVariable):
            self.parents['n'] = n

    def _process_parameters(self, **kwargs):
        """Process parameters for the binomial distribution."""
        # Parameters are handled in __init__
        pass

    def pdf(self, x, **kwargs):
        """
        Probability mass function for binomial distribution.

        Parameters:
            x: int or array-like
                Number of successes.
            **kwargs: dict
                Additional parameters (ignored for binomial).

        Returns:
            float or array-like: Probability mass at the given point(s).
        """
        new_kwargs = {
            'n': self.n,
            'p': self.p
        }

        new_kwargs.update(kwargs)

        return binom.pmf(x, **new_kwargs)

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability mass function.

        Parameters:
            x: int or array-like
                Number of successes.
            **kwargs: dict
                Will override parameters n and p.

        Returns:
            float or array-like: Log probability mass at the given point(s).
        """
        new_kwargs = {
            'n': self.n,
            'p': self.p
        }

        new_kwargs.update(kwargs)

        return binom.logpmf(x, **new_kwargs)

    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the binomial distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Additional parameters (ignored for binomial).

        Returns:
            int or array-like: Random samples from the distribution.
        """
        new_kwargs = {
            'n': self.n,
            'p': self.p
        }

        new_kwargs.update(kwargs)

        return binom.rvs(size=size, **new_kwargs)

    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function.

        Parameters:
            x: int or array-like
                Number of successes.
            **kwargs: dict
                Additional parameters (ignored for binomial).

        Returns:
            float or array-like: Cumulative probability at the given point(s).
        """
        new_kwargs = {
            'n': self.n,
            'p': self.p
        }

        new_kwargs.update(kwargs)

        return binom.cdf(x, **new_kwargs)

    def __repr__(self):
        """String representation of the binomial random variable."""
        if self.name:
            return f"Binomial(name='{self.name}', n={self.n}, p={self.p})"
        else:
            return f"Binomial(n={self.n}, p={self.p})"

    def __str__(self):
        """String representation of the binomial random variable."""
        return self.__repr__()
