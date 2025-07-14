import numpy as np
from ..linx.ds import BayesianNetwork as BN
from ..linx.random.beta import Beta
from ..linx.random.binomial import Binomial
from ..linx.random.deterministic import Deterministic
from ..linx.random.logistic import Logistic


def test_debug_bayesian_network_sampling():
    """
    Simple debug test to understand the issue with Bayesian network sampling
    when using Deterministic and Logistic nodes.
    """
    print("=== Debug Test: Bayesian Network with Deterministic/Logistic Nodes ===")

    # Create a simple Bayesian network
    bn = BN()

    # Add a Beta prior
    beta_prior = Beta(name='beta_param', a=2, b=3)
    bn.add_node(beta_prior)
    print(f"Added beta_prior: {beta_prior.name}")

    # Add a Deterministic node - using identity function
    def identity_func(beta_param):
        return beta_param

    det_node = Deterministic(name='deterministic_node', callable_func=identity_func, beta_param=beta_prior)
    bn.add_node(det_node)
    print(f"Added det_node: {det_node.name}")

    # Add a Logistic node - using identity function
    def identity_func2(deterministic_node):
        return deterministic_node

    logistic_node = Logistic(name='logistic_node', callable_func=identity_func2, deterministic_node=det_node)
    bn.add_node(logistic_node)
    print(f"Added logistic_node: {logistic_node.name}")

    # Add a Binomial likelihood
    likelihood = Binomial(name='observation', n=10, p=logistic_node)
    bn.add_node(likelihood)
    print(f"Added likelihood: {likelihood.name}")

    # Print network structure
    print(f"\nNetwork random_variables keys: {list(bn.random_variables.keys())}")
    print(f"Network children: {bn.children}")
    print(f"Network nodes: {bn.get_nodes()}")

    # Try to get topological sort
    try:
        topo_sort = bn.topological_sort()
        print(f"Topological sort: {topo_sort}")
    except Exception as e:
        print(f"Error in topological sort: {e}")

    # Try to sample
    try:
        print("\nAttempting to sample from network...")
        sample = bn.sample()
        print(f"Sample successful: {sample}")
    except Exception as e:
        print(f"Error in sampling: {e}")
        import traceback
        traceback.print_exc()


def test_debug_deterministic_parents():
    """
    Test to understand how Deterministic nodes handle parents.
    """
    print("\n=== Debug Test: Deterministic Parents ===")

    beta_prior = Beta(name='beta_param', a=2, b=3)

    def identity_func(beta_param):
        return beta_param

    det_node = Deterministic(name='deterministic_node', callable_func=identity_func, beta_param=beta_prior)

    print(f"Deterministic node name: {det_node.name}")
    print(f"Deterministic node parents: {det_node.get_parents()}")
    print(f"Deterministic node callable_func: {det_node.callable_func}")
    print(f"Parent names: {list(det_node.parents.keys())}")


def test_debug_logistic_parents():
    """
    Test to understand how Logistic nodes handle parents.
    """
    print("\n=== Debug Test: Logistic Parents ===")

    beta_prior = Beta(name='beta_param', a=2, b=3)

    def identity_func(beta_param):
        return beta_param

    det_node = Deterministic(name='deterministic_node', callable_func=identity_func, beta_param=beta_prior)

    def identity_func2(deterministic_node):
        return deterministic_node

    logistic_node = Logistic(name='logistic_node', callable_func=identity_func2, deterministic_node=det_node)

    print(f"Logistic node name: {logistic_node.name}")
    print(f"Logistic node parents: {logistic_node.get_parents()}")
    print(f"Logistic node callable_func: {logistic_node.callable_func}")
    print(f"Parent names: {list(logistic_node.parents.keys())}")


if __name__ == "__main__":
    test_debug_deterministic_parents()
    test_debug_logistic_parents()
    test_debug_bayesian_network_sampling()
