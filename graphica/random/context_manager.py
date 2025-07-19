"""
Context manager for Bayesian Network automatic node addition.
"""

import threading
import uuid

# Thread-local storage for the current Bayesian network
_local = threading.local()


def set_current_network(network):
    """
    Set the current Bayesian network for the current thread.

    Parameters:
        network: BayesianNetwork
            The Bayesian network to set as current.
    """
    _local.current_network = network


def get_current_network():
    """
    Get the current Bayesian network for the current thread.

    Returns:
        BayesianNetwork or None: The current Bayesian network, or None if not set.
    """
    return getattr(_local, "current_network", None)


def clear_current_network():
    """
    Clear the current Bayesian network for the current thread.
    """
    if hasattr(_local, "current_network"):
        delattr(_local, "current_network")


def generate_variable_name():
    """
    Generate a unique variable name.

    Returns:
        str: A unique variable name.
    """
    return f"var_{str(uuid.uuid4())[:8]}"


def add_random_variable_to_current_network(random_variable):
    """
    Add a random variable to the current Bayesian network if one is set.

    Parameters:
        random_variable: RandomVariable
            The random variable to add to the current network.

    Returns:
        bool: True if the variable was added to a network, False otherwise.
    """
    network = get_current_network()
    if network is not None:
        # Generate a name if the variable doesn't have one
        if random_variable.name is None:
            random_variable.name = generate_variable_name()

        network.add_node(random_variable)
        return True
    return False
