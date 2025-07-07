"""
Normal distribution implementation.
"""
import numpy as np
from .random_variable import RandomVariable


class Normal(RandomVariable):
    """
    Normal (Gaussian) distribution.

    Parameters:
        mean: float, default=0.0
            Mean of the normal distribution.
        std: float, default=1.0
            Standard deviation of the normal distribution.
    """

    def _process_parameters(self, mean=0.0, std=1.0, **kwargs):
        """
        Process parameters for normal distribution.

        Parameters:
            mean: float
                Mean of the distribution.
            std: float
                Standard deviation of the distribution.
        """
        if std <= 0:
            raise ValueError("Standard deviation must be positive")

        self.mean = mean
        self.std = std
        self.var = std ** 2  # Variance

    def pdf(self, x, **kwargs):
        """
        Probability density function of the normal distribution.

        f(x) = (1 / (σ * √(2π))) * exp(-0.5 * ((x - μ) / σ)²)

        Parameters:
            x: array-like
                Points at which to evaluate the PDF.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: PDF values at the given points.
        """
        x = np.asarray(x)
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        log(f(x)) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²

        Parameters:
            x: array-like
                Points at which to evaluate the log PDF.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: Log PDF values at the given points.
        """
        x = np.asarray(x)
        try :

            return -0.5 * np.log(2 * np.pi) - np.log(self.std) - 0.5 * ((x - self.mean) / self.std) ** 2
        except Exception as e:
            import pdb; pdb.set_trace()


    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the normal distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: Random samples from the normal distribution.
        """
        return np.random.normal(self.mean, self.std, size=size)

    def cdf(self, x):
        """
        Cumulative distribution function.

        Parameters:
            x: array-like
                Points at which to evaluate the CDF.

        Returns:
            array-like: CDF values at the given points.
        """
        x = np.asarray(x)
        # Use scipy's normal CDF for accuracy
        from scipy import stats
        return stats.norm.cdf(x, loc=self.mean, scale=self.std)

    def __repr__(self):
        """String representation of the normal distribution."""
        if self.name:
            return f"Normal(name='{self.name}', mean={self.mean}, std={self.std})"
        else:
            return f"Normal(mean={self.mean}, std={self.std})"

    def __str__(self):
        """String representation of the normal distribution."""
        return self.__repr__()
