"""
RandomVariable abstract class for probability distributions.
"""

from abc import ABC, abstractmethod
from .context_manager import add_random_variable_to_current_network


class RandomVariable(ABC):
    """
    Abstract base class for random variables.

    This class defines the interface that all random variables must implement.
    Random variables can have probability density functions (pdf), log probability
    density functions (logpdf), and sampling capabilities.
    """

    # pylint: disable=unused-argument
    def __init__(self, name=None, **kwargs):
        """
        Initialize random variable.

        Parameters:
            name: str, optional
                Name of the random variable.
            **kwargs: dict
                Additional parameters for the specific distribution.
        """
        self.name = name
        self.parents = {}

        # Automatically add to current Bayesian network if one is set
        try:
            add_random_variable_to_current_network(self)
        except ImportError:
            # If context_manager is not available, ignore
            pass

    def set_parents(self, parents):
        """
        Set parent random variables.

        Parameters:
            parents: dict[str, RandomVariable] or list[RandomVariable]
                Dictionary mapping parent names to RandomVariable objects,
                or list of RandomVariable objects (will use their names as keys).
        """
        if isinstance(parents, dict):
            self.parents = parents
        elif isinstance(parents, list):
            # Convert list to dict using parent names as keys
            self.parents = {parent.name: parent for parent in parents if parent.name}
        else:
            raise TypeError("parents must be a dict or list")

    def get_parents(self):
        """
        Get parent random variables.

        Returns:
            dict[str, RandomVariable]
                Dictionary mapping parent names to RandomVariable objects.
        """
        return self.parents

    @abstractmethod
    def pdf(self, x, **kwargs):
        """
        Probability density function.

        Parameters:
            x: array-like
                Points at which to evaluate the probability density function.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: Probability density values at the given points.
        """

    @abstractmethod
    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability density function.

        Parameters:
            x: array-like
                Points at which to evaluate the log probability density function.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: Log probability density values at the given points.
        """

    @abstractmethod
    def sample(self, size=None, **kwargs):
        """
        Generate random samples from the distribution.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then
                m * n * k samples are drawn. If size is None (default),
                a single value is returned.
            **kwargs: dict
                Additional parameters (e.g., parent values for conditional distributions).

        Returns:
            array-like: Random samples from the distribution.
        """

    def perturb(self, current_value, **kwargs):
        """
        Perturb the current value according to the distribution's characteristics.

        Parameters:
            current_value: float
                The current value to perturb.
            **kwargs: dict
                Perturbation parameters specific to the distribution.

        Returns:
            float: The perturbed value.
        """
        raise NotImplementedError(
            f"perturb method not implemented for {self.__class__.__name__}"
        )

    def __repr__(self):
        """String representation of the random variable."""
        if self.name:
            return f"{self.__class__.__name__}(name='{self.name}')"

        return f"{self.__class__.__name__}()"

    def __str__(self):
        """String representation of the random variable."""
        return self.__repr__()
