"""
Simplified Bayesian Network Example

This example demonstrates how to create a simple Bayesian network
with continuous random variables and perform inference.
"""

import numpy as np
from graphica.bayesian_network import BayesianNetwork
from graphica.random.normal import Normal
from graphica.random.uniform import Uniform
from graphica.conditional_probability_table import ConditionalProbabilityTable as CPT
from graphica.inference.metropolis_hastings import MetropolisHastings
from graphica.query import Query


def main():
    """Demonstrate the simplified Bayesian network interface."""

    # Create a Bayesian network with mixed random variables
    bn = BayesianNetwork()

    # Add a Normal random variable (prior)
    normal_rv = Normal(name="X", mu=0, sigma=1)
    bn.add_node(normal_rv)

    # Add a Uniform random variable (prior)
    uniform_rv = Uniform(name="Y", low=0, high=1)
    bn.add_node(uniform_rv)

    # Add a CPT that depends on both continuous variables
    # This represents a discrete variable Z that depends on X and Y
    cpt_z = CPT(
        table=[
            {"X": 0, "Y": 0, "Z": 0, "value": 0.8},
            {"X": 0, "Y": 0, "Z": 1, "value": 0.2},
            {"X": 0, "Y": 1, "Z": 0, "value": 0.6},
            {"X": 0, "Y": 1, "Z": 1, "value": 0.4},
            {"X": 1, "Y": 0, "Z": 0, "value": 0.3},
            {"X": 1, "Y": 0, "Z": 1, "value": 0.7},
            {"X": 1, "Y": 1, "Z": 0, "value": 0.1},
            {"X": 1, "Y": 1, "Z": 1, "value": 0.9},
        ],
        outcomes=["Z"],
        givens=["X", "Y"],
        name="Z",
    )
    bn.add_node(cpt_z)

    print("Bayesian Network created with mixed random variables:")
    print(f"Nodes: {list(bn.get_random_variables().keys())}")
    print(f"Graph structure: {bn.get_nodes()}")
    print()

    # Sample from the network
    print("Sampling from the network:")
    for i in range(5):
        particle = bn.sample()
        print(
            f"Sample {i+1}: X={particle.get_value('X'):.3f}, "
            f"Y={particle.get_value('Y'):.3f}, Z={particle.get_value('Z')}"
        )
    print()

    # Demonstrate Metropolis-Hastings sampling with conditioning
    print("Metropolis-Hastings sampling with conditioning on Z=1:")

    # Define a simple transition function (random walk)
    def transition_function(particle):
        """Simple random walk transition function."""
        new_particle = particle.copy()

        # Perturb continuous variables
        if new_particle.has_variable("X"):
            x_val = new_particle.get_value("X")
            new_particle.set_value("X", x_val + np.random.normal(0, 0.1))

        if new_particle.has_variable("Y"):
            y_val = new_particle.get_value("Y")
            new_y = y_val + np.random.normal(0, 0.1)
            # Keep Y in [0, 1]
            new_particle.set_value("Y", np.clip(new_y, 0, 1))

        return new_particle

    # Create query conditioning on Z=1
    query = Query(outcomes=["X", "Y"], givens=[{"Z": 1}])

    # Create Metropolis-Hastings sampler
    mh = MetropolisHastings(
        network=bn, query=query, transition_function=transition_function
    )

    # Sample with Metropolis-Hastings
    samples = mh.sample(n=100, burn_in=50)

    print(f"Generated {len(samples)} samples")
    print("First 5 samples:")
    for i, sample in enumerate(samples[:5]):
        print(
            f"  Sample {i+1}: X={sample.get_value('X'):.3f}, "
            f"Y={sample.get_value('Y'):.3f}"
        )

    # Calculate acceptance rate
    accepted = sum(1 for s in samples if s.is_accepted())
    print(f"Acceptance rate: {accepted/len(samples):.3f}")

    # Show that all random variables are treated uniformly
    print("\nAll nodes are RandomVariables:")
    for name, rv in bn.get_random_variables().items():
        print(f"  {name}: {type(rv).__name__}")
        print(f"    - Has pdf: {hasattr(rv, 'pdf')}")
        print(f"    - Has logpdf: {hasattr(rv, 'logpdf')}")
        print(f"    - Has sample: {hasattr(rv, 'sample')}")


if __name__ == "__main__":
    main()
