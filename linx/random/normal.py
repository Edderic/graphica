"""
Normal distribution implementation.
"""
import numpy as np
from .random_variable import RandomVariable


class Normal(RandomVariable):
    """
    Normal (Gaussian) distribution.

    Parameters:
        name: str, optional
            Name of the random variable.
        mean: float or RandomVariable, default=0.0
            Mean of the normal distribution.
        std: float or RandomVariable, default=1.0
            Standard deviation of the normal distribution.
    """

    def __init__(self, name=None, mean=0.0, std=1.0, **kwargs):
        """
        Initialize Normal random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            mean: float or RandomVariable
                Mean of the distribution.
            std: float or RandomVariable
                Standard deviation of the distribution.
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std

        # Validate parameters and set parents
        if isinstance(mean, RandomVariable):
            self.parents['mean'] = mean
        elif not isinstance(std, RandomVariable) and std <= 0:
            raise ValueError("Standard deviation must be positive")

        if isinstance(std, RandomVariable):
            self.parents['std'] = std
        elif not isinstance(mean, RandomVariable) and std <= 0:
            raise ValueError("Standard deviation must be positive")

        # Set variance for non-RandomVariable std
        if not isinstance(std, RandomVariable):
            self.var = std ** 2

    def _process_parameters(self, **kwargs):
        """Process parameters for normal distribution."""
        # Parameters are handled in __init__
        pass

    def pdf(self, x, **kwargs):
        """
        Probability density function of the normal distribution.

        f(x) = (1 / (σ * √(2π))) * exp(-0.5 * ((x - μ) / σ)²)

        Parameters:
            x: array-like
                Points at which to evaluate the PDF.
            **kwargs: dict
                Will override parameters mean and std.

        Returns:
            array-like: PDF values at the given points.
        """
        new_kwargs = {
            'mean': self.mean,
            'std': self.std
        }
        new_kwargs.update(kwargs)

        mean = new_kwargs['mean']
        std = new_kwargs['std']

        x = np.asarray(x)
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        log(f(x)) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²

        Parameters:
            x: array-like
                Points at which to evaluate the log PDF.
            **kwargs: dict
                Will override parameters mean and std.

        Returns:
            array-like: Log PDF values at the given points.
        """
        new_kwargs = {
            'mean': self.mean,
            'std': self.std
        }
        new_kwargs.update(kwargs)

        mean = new_kwargs['mean']
        std = new_kwargs['std']

        x = np.asarray(x)
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((x - mean) / std) ** 2

    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the normal distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Will override parameters mean and std.

        Returns:
            array-like: Random samples from the normal distribution.
        """
        new_kwargs = {
            'mean': self.mean,
            'std': self.std
        }
        new_kwargs.update(kwargs)

        mean = new_kwargs['mean']
        std = new_kwargs['std']

        return np.random.normal(mean, std, size=size)

    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function.

        Parameters:
            x: array-like
                Points at which to evaluate the CDF.
            **kwargs: dict
                Will override parameters mean and std.

        Returns:
            array-like: CDF values at the given points.
        """
        new_kwargs = {
            'mean': self.mean,
            'std': self.std
        }
        new_kwargs.update(kwargs)

        mean = new_kwargs['mean']
        std = new_kwargs['std']

        x = np.asarray(x)
        # Use scipy's normal CDF for accuracy
        from scipy import stats
        return stats.norm.cdf(x, loc=mean, scale=std)

    def __repr__(self):
        """String representation of the normal distribution."""
        if self.name:
            return f"Normal(name='{self.name}', mean={self.mean}, std={self.std})"
        else:
            return f"Normal(mean={self.mean}, std={self.std})"

    def __str__(self):
        """String representation of the normal distribution."""
        return self.__repr__()
