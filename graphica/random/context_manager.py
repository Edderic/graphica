"""
Context manager helper for Random Variables in Bayesian Networks.
"""


class RandomVariableContext:
    """
    Context manager helper for Random Variables.

    This class allows Random Variables to be used in context managers
    with Bayesian Networks.
    """

    def __init__(self, network, rv):
        """
        Initialize the context.

        Parameters:
            network: BayesianNetwork
                The Bayesian network to add the random variable to.
            rv: RandomVariable
                The random variable to add.
        """
        self.network = network
        self.rv = rv

    def __enter__(self):
        """Context entry - add the random variable to the network."""
        if self.rv.name:
            self.network.random_variables[self.rv.name] = self.rv
            self.network.add_node(self.rv.name)
        return self.rv

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context exit - nothing to do here."""
        pass


def add_to_network(network, rv):
    """
    Add a random variable to a network and return it for context manager use.

    Parameters:
        network: BayesianNetwork
            The Bayesian network to add the random variable to.
        rv: RandomVariable
            The random variable to add.

    Returns:
        RandomVariable: The random variable (for chaining).
    """
    if rv.name:
        network.random_variables[rv.name] = rv
        network.add_node(rv.name)
    return rv
