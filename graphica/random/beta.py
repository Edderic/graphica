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
        a: float or RandomVariable
            First shape parameter (a > 0).
        b: float or RandomVariable
            Second shape parameter (beta > 0).
    """

    def __init__(self, name=None, a=1.0, b=1.0, **kwargs):
        """
        Initialize Beta random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            a: float or RandomVariable
                First shape parameter (a > 0).
            b: float or RandomVariable
                Second shape parameter (beta > 0).
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.a = a
        self.b = b

        # Validate parameters and set parents
        if isinstance(a, RandomVariable):
            self.parents["a"] = a
        elif not isinstance(b, RandomVariable) and a <= 0:
            raise ValueError("a must be positive")

        if isinstance(b, RandomVariable):
            self.parents["b"] = b
        elif not isinstance(a, RandomVariable) and b <= 0:
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
                Will override parameters a and b.

        Returns:
            float or array-like: Probability density at the given point(s).
        """
        new_kwargs = {"a": self.a, "b": self.b}
        new_kwargs.update(kwargs)

        a = new_kwargs["a"]
        b = new_kwargs["b"]

        return beta.pdf(x, a=a, b=b)

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        Parameters:
            x: float or array-like
                Points in [0, 1] at which to evaluate the log PDF.
            **kwargs: dict
                Will override parameters a and b.

        Returns:
            float or array-like: Log probability density at the given point(s).
        """
        new_kwargs = {"a": self.a, "b": self.b}
        new_kwargs.update(kwargs)

        a = new_kwargs["a"]
        b = new_kwargs["b"]

        return beta.logpdf(x, a=a, b=b)

    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the beta distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Will override parameters a and b.

        Returns:
            float or array-like: Random samples from the distribution.
        """
        new_kwargs = {"a": self.a, "b": self.b}
        new_kwargs.update(kwargs)

        a = new_kwargs["a"]
        b = new_kwargs["b"]

        return beta.rvs(a, b, size=size)

    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function.

        Parameters:
            x: float or array-like
                Points in [0, 1] at which to evaluate the CDF.
            **kwargs: dict
                Will override parameters a and b.

        Returns:
            float or array-like: Cumulative probability at the given point(s).
        """
        new_kwargs = {"a": self.a, "b": self.b}
        new_kwargs.update(kwargs)

        a = new_kwargs["a"]
        b = new_kwargs["b"]

        return beta.cdf(x, a=a, b=b)

    def __repr__(self):
        """String representation of the beta random variable."""
        if self.name:
            return f"Beta(name='{self.name}', a={self.a}, b={self.b})"
        else:
            return f"Beta(a={self.a}, b={self.b})"

    def perturb(self, current_value, low=-0.1, high=0.1, **kwargs):
        """
        Perturb the current value by adding uniform noise and clipping to [0, 1].

        Parameters:
            current_value: float
                The current value to perturb.
            low: float, default=-0.1
                Lower bound of uniform noise.
            high: float, default=0.1
                Upper bound of uniform noise.
            **kwargs: dict
                Additional parameters (ignored).

        Returns:
            float: The perturbed value, clipped to [0, 1].
        """
        noise = np.random.uniform(low, high)
        return np.clip(current_value + noise, 0, 1)

    def __str__(self):
        """String representation of the beta random variable."""
        return self.__repr__()
