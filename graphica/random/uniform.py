"""
Uniform distribution implementation.
"""

import numpy as np
from .random_variable import RandomVariable


class Uniform(RandomVariable):
    """
    Uniform distribution.

    Parameters:
        name: str, optional
            Name of the random variable.
        low: float or RandomVariable
            Lower bound of the uniform distribution.
        high: float or RandomVariable
            Upper bound of the uniform distribution.
    """

    def __init__(self, name=None, low=0.0, high=1.0, **kwargs):
        """
        Initialize Uniform random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            low: float or RandomVariable
                Lower bound of the distribution.
            high: float or RandomVariable
                Upper bound of the distribution.
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.low = low
        self.high = high

        # Validate parameters and set parents
        if isinstance(low, RandomVariable):
            self.parents["low"] = low
        elif not isinstance(high, RandomVariable) and low >= high:
            raise ValueError("Lower bound must be less than upper bound")

        if isinstance(high, RandomVariable):
            self.parents["high"] = high
        elif not isinstance(low, RandomVariable) and low >= high:
            raise ValueError("Lower bound must be less than upper bound")

    def _process_parameters(self, **kwargs):
        """Process parameters for uniform distribution."""
        # Parameters are handled in __init__
        pass

    def pdf(self, x, **kwargs):
        """
        Probability density function of the uniform distribution.

        f(x) = 1/(high-low) for low <= x <= high, 0 otherwise.

        Parameters:
            x: array-like
                Points at which to evaluate the PDF.
            **kwargs: dict
                Will override parameters low and high.

        Returns:
            array-like: PDF values at the given points.
        """
        new_kwargs = {"low": self.low, "high": self.high}
        new_kwargs.update(kwargs)

        low = new_kwargs["low"]
        high = new_kwargs["high"]
        width = high - low

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= low) & (x <= high)
        result[mask] = 1.0 / width
        return result

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        log(f(x)) = -log(high-low) for low <= x <= high, -inf otherwise.

        Parameters:
            x: array-like
                Points at which to evaluate the log PDF.
            **kwargs: dict
                Will override parameters low and high.

        Returns:
            array-like: Log PDF values at the given points.
        """
        new_kwargs = {"low": self.low, "high": self.high}
        new_kwargs.update(kwargs)

        low = new_kwargs["low"]
        high = new_kwargs["high"]
        width = high - low

        x = np.asarray(x)
        result = np.full_like(x, -np.inf, dtype=float)
        mask = (x >= low) & (x <= high)
        result[mask] = -np.log(width)
        return result

    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the uniform distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Will override parameters low and high.

        Returns:
            array-like: Random samples from the distribution.
        """
        new_kwargs = {"low": self.low, "high": self.high}
        new_kwargs.update(kwargs)

        low = new_kwargs["low"]
        high = new_kwargs["high"]

        return np.random.uniform(low, high, size=size)

    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function of the uniform distribution.

        Parameters:
            x: array-like
                Points at which to evaluate the CDF.
            **kwargs: dict
                Will override parameters low and high.

        Returns:
            array-like: CDF values at the given points.
        """
        new_kwargs = {"low": self.low, "high": self.high}
        new_kwargs.update(kwargs)

        low = new_kwargs["low"]
        high = new_kwargs["high"]
        width = high - low

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= low) & (x <= high)
        result[mask] = (x[mask] - low) / width
        result[x > high] = 1.0
        return result

    def __repr__(self):
        """String representation of the uniform distribution."""
        if self.name:
            return f"Uniform(name='{self.name}', low={self.low}, high={self.high})"
        else:
            return f"Uniform(low={self.low}, high={self.high})"

    def perturb(self, current_value, low=-0.1, high=0.1, **kwargs):
        """
        Perturb the current value by adding uniform noise.

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
            float: The perturbed value.
        """
        noise = np.random.uniform(low, high)
        return current_value + noise

    def __str__(self):
        """String representation of the uniform distribution."""
        return self.__repr__()
