"""
Gamma distribution implementation.
"""
import numpy as np
from .random_variable import RandomVariable

class Gamma(RandomVariable):
    """
    Gamma distribution.
    
    Parameters:
        name: str, optional
            Name of the random variable.
        shape: float or RandomVariable
            Shape parameter (k or alpha), must be > 0.
        scale: float or RandomVariable, default=1.0
            Scale parameter (theta), must be > 0.
    """
    
    def __init__(self, name=None, shape=1.0, scale=1.0, **kwargs):
        """
        Initialize Gamma random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            shape: float or RandomVariable
                Shape parameter (k or alpha), must be > 0.
            scale: float or RandomVariable, default=1.0
                Scale parameter (theta), must be > 0.
            **kwargs: dict
                Additional parameters passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self.shape = shape
        self.scale = scale

        # Validate parameters and set parents
        if isinstance(shape, RandomVariable):
            self.parents['shape'] = shape
        elif not isinstance(scale, RandomVariable) and shape <= 0:
            raise ValueError("Shape parameter must be positive")

        if isinstance(scale, RandomVariable):
            self.parents['scale'] = scale
        elif not isinstance(shape, RandomVariable) and scale <= 0:
            raise ValueError("Scale parameter must be positive")

    def _process_parameters(self, **kwargs):
        """Process parameters for gamma distribution."""
        # Parameters are handled in __init__
        pass
    
    def pdf(self, x, **kwargs):
        """
        Probability density function of the gamma distribution.

        Parameters:
            x: array-like
                Points at which to evaluate the PDF.
            **kwargs: dict
                Will override parameters shape and scale.

        Returns:
            array-like: PDF values at the given points.
        """
        new_kwargs = {
            'shape': self.shape,
            'scale': self.scale
        }
        new_kwargs.update(kwargs)

        shape = new_kwargs['shape']
        scale = new_kwargs['scale']

        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.pdf(x, a=shape, scale=scale)
    
    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        Parameters:
            x: array-like
                Points at which to evaluate the log PDF.
            **kwargs: dict
                Will override parameters shape and scale.

        Returns:
            array-like: Log PDF values at the given points.
        """
        new_kwargs = {
            'shape': self.shape,
            'scale': self.scale
        }
        new_kwargs.update(kwargs)

        shape = new_kwargs['shape']
        scale = new_kwargs['scale']

        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.logpdf(x, a=shape, scale=scale)
    
    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the gamma distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Will override parameters shape and scale.

        Returns:
            array-like: Random samples from the distribution.
        """
        new_kwargs = {
            'shape': self.shape,
            'scale': self.scale
        }
        new_kwargs.update(kwargs)

        shape = new_kwargs['shape']
        scale = new_kwargs['scale']

        return np.random.gamma(shape=shape, scale=scale, size=size)
    
    def cdf(self, x, **kwargs):
        """
        Cumulative distribution function of the gamma distribution.

        Parameters:
            x: array-like
                Points at which to evaluate the CDF.
            **kwargs: dict
                Will override parameters shape and scale.

        Returns:
            array-like: CDF values at the given points.
        """
        new_kwargs = {
            'shape': self.shape,
            'scale': self.scale
        }
        new_kwargs.update(kwargs)

        shape = new_kwargs['shape']
        scale = new_kwargs['scale']

        from scipy.stats import gamma
        x = np.asarray(x)
        return gamma.cdf(x, a=shape, scale=scale)
    
    def __repr__(self):
        if self.name:
            return f"Gamma(name='{self.name}', shape={self.shape}, scale={self.scale})"
        else:
            return f"Gamma(shape={self.shape}, scale={self.scale})"
    
    def __str__(self):
        return self.__repr__() 