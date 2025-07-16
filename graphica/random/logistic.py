"""
Logistic random variable implementation.
"""

import numpy as np
from .deterministic import Deterministic


class Logistic(Deterministic):
    """
    Logistic random variable that applies the logistic function to a deterministic transformation.

    Parameters:
        callable_func: callable
            A function that takes keyword arguments and returns a value.
        **kwargs: dict
            Keyword arguments to pass to the callable function.
            RandomVariable instances will be set as parents.
    """

    def __init__(self, callable_func, name=None, **kwargs):
        """
        Initialize Logistic random variable.

        Parameters:
            callable_func: callable
                A function that takes keyword arguments and returns a value.
            name: str, optional
                Name of the random variable. If None, a UUID will be generated.
            **kwargs: dict
                Keyword arguments to pass to the callable function.
                RandomVariable instances will be set as parents.
        """
        super().__init__(callable_func, name=name, **kwargs)

    def _logistic_function(self, x):
        """
        Apply the logistic function: 1 / (1 + exp(-x))

        Parameters:
            x: array-like
                Input values.

        Returns:
            array-like: Logistic transformation of the input.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def sample(self, size=None, **kwargs):
        """
        Generate samples by evaluating the callable function and applying the logistic transformation.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Additional parameters that will override fixed parameters.

        Returns:
            array-like: Result of evaluating the callable function with logistic transformation.
        """
        # Get the result from the parent class (Constant)
        result = super().sample(size=None, **kwargs)

        # Apply logistic transformation
        logistic_result = self._logistic_function(result)

        # Handle size parameter
        if size is not None:
            if np.isscalar(logistic_result):
                return np.full(size, logistic_result)
            else:
                # For non-scalar results, repeat the result
                result_array = np.asarray(logistic_result)
                return np.tile(result_array, size)

        return logistic_result

    def __repr__(self):
        """String representation of the logistic random variable."""
        if self.name:
            return (
                f"Logistic(name='{self.name}', callable={self.callable_func.__name__})"
            )
        else:
            return f"Logistic(callable={self.callable_func.__name__})"

    def __str__(self):
        """String representation of the logistic random variable."""
        return self.__repr__()
