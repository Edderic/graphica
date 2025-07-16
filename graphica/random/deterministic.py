"""
Deterministic random variable implementation.
"""

import numpy as np
import uuid
from .random_variable import RandomVariable


class Deterministic(RandomVariable):
    """
    Deterministic random variable that represents a deterministic transformation.

    Parameters:
        callable_func: callable
            A function that takes keyword arguments and returns a value.
        name: str, optional
            Name of the random variable. If None, a UUID will be generated.
        **kwargs: dict
            Keyword arguments to pass to the callable function.
            RandomVariable instances will be set as parents.
    """

    def __init__(self, callable_func, name=None, **kwargs):
        """
        Initialize Deterministic random variable.

        Parameters:
            callable_func: callable
                A function that takes keyword arguments and returns a value.
            name: str, optional
                Name of the random variable. If None, a UUID will be generated.
            **kwargs: dict
                Keyword arguments to pass to the callable function.
                RandomVariable instances will be set as parents.
        """
        # Generate UUID if no name provided
        if name is None:
            name = str(uuid.uuid4())

        super().__init__(name=name)
        self.callable_func = callable_func
        self.fixed_params = {}
        self.parents = {}

        # Separate RandomVariable instances from fixed parameters
        for key, value in kwargs.items():
            if isinstance(value, RandomVariable):
                self.parents[key] = value
            else:
                self.fixed_params[key] = value

    def _process_parameters(self, **kwargs):
        """Process parameters for deterministic distribution."""
        # Parameters are handled in __init__
        pass

    def pdf(self, x, **kwargs):
        """
        Probability density function. Always returns 1 for deterministic variables.

        Parameters:
            x: array-like
                Points at which to evaluate the PDF.
            **kwargs: dict
                Additional parameters (unused for deterministic).

        Returns:
            array-like: PDF values (always 1).
        """
        x = np.asarray(x)
        return np.ones_like(x, dtype=float)

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function. Always returns 0 for deterministic variables.

        Parameters:
            x: array-like
                Points at which to evaluate the log PDF.
            **kwargs: dict
                Additional parameters (unused for deterministic).

        Returns:
            array-like: Log PDF values (always 0).
        """
        x = np.asarray(x)
        return np.zeros_like(x, dtype=float)

    def sample(self, size=None, **kwargs):
        """
        Generate samples by evaluating the callable function.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Additional parameters that will override fixed parameters.

        Returns:
            array-like: Result of evaluating the callable function.
        """
        # Merge fixed parameters with provided kwargs (kwargs take precedence)
        params = self.fixed_params.copy()
        params.update(kwargs)

        # Check if all required parameters are available
        import inspect

        sig = inspect.signature(self.callable_func)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
        ]

        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        try:
            result = self.callable_func(**params)
        except Exception as e:
            raise ValueError(f"Error calling function with parameters {params}: {e}")

        # Handle size parameter
        if size is not None:
            if np.isscalar(result):
                return np.full(size, result)
            else:
                # For non-scalar results, repeat the result
                result_array = np.asarray(result)
                if isinstance(size, int):
                    return np.tile(result_array, (size,) + (1,) * result_array.ndim)
                else:
                    return np.tile(result_array, size)

        return result

    def get_parents(self):
        """
        Get parent random variables.

        Returns:
            dict[str, RandomVariable]
                Dictionary mapping parent names to RandomVariable objects.
        """
        return self.parents

    def __repr__(self):
        """String representation of the deterministic random variable."""
        if self.name:
            return f"Deterministic(name='{self.name}', callable={self.callable_func.__name__})"
        else:
            return f"Deterministic(callable={self.callable_func.__name__})"

    def perturb(self, current_value, **kwargs):
        """
        Perturb the current value by adding small uniform noise.
        For Deterministic nodes, we add a small perturbation to allow for exploration.

        Parameters:
            current_value: float
                The current value to perturb.
            **kwargs: dict
                Additional parameters (ignored).

        Returns:
            float: The perturbed value.
        """
        # Add small uniform noise for exploration
        noise = np.random.uniform(-0.01, 0.01)
        return current_value + noise

    def __str__(self):
        """String representation of the deterministic random variable."""
        return self.__repr__()
