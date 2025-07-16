"""
Example demonstrating Random Variables in Bayesian Networks.
"""

import numpy as np
from ..ds import BayesianNetwork, Normal, Uniform, Query, MetropolisHastings, Particle


def transition_function(particle):
    """Simple transition function for Metropolis-Hastings."""
    new_particle = particle.copy()

    # Update mu with random walk
    if particle.has_variable("mu"):
        current_mu = particle.get_value("mu")
        new_mu = current_mu + np.random.normal(0, 1)
        new_particle.set_value("mu", new_mu)

    # Update sigma with random walk (ensure positivity)
    if particle.has_variable("sigma"):
        current_sigma = particle.get_value("sigma")
        new_sigma = current_sigma + np.random.normal(0, 1)
        new_sigma = max(0.1, new_sigma)  # Ensure positive
        new_particle.set_value("sigma", new_sigma)

    return new_particle


def run_random_variable_example():
    """Run the random variable Bayesian network example."""
    print("Random Variable Bayesian Network Example")
    print("=" * 50)

    # Example 1: Using Random Variables with context manager
    print("\n1. Random Variables with context manager:")

    with BayesianNetwork() as bn:
        # Create random variables
        mu = Normal(name="mu", mean=1.0, std=2.0)
        sigma = Uniform(name="sigma", low=0.1, high=10.0)

        # Add them to the network
        bn.add_random_variable(mu)
        bn.add_random_variable(sigma)

        print(f"Network has {len(bn.random_variables)} random variables")
        for name, rv in bn.random_variables.items():
            print(f"  - {name}: {rv}")

    print("\n2. Metropolis-Hastings sampling:")

    # Create a simple network for sampling
    bn = BayesianNetwork()
    mu = Normal(name="mu", mean=1.0, std=5.0)
    sigma = Uniform(name="sigma", low=0.1, high=3.0)
    X = Normal(name="X", mean=mu, sigma=sigma)

    bn.add_random_variable(mu)
    bn.add_random_variable(sigma)
    bn.add_random_variable(X)

    # Create a query conditioning on some observation
    query = Query(outcomes=[], givens=[{"X": 1.5}])  # Observe X = 1.5

    # Create sampler
    sampler = MetropolisHastings(
        network=bn, query=query, transition_function=transition_function
    )

    # Sample (with smaller number for demonstration)
    print("Sampling 100 particles...")
    particles = sampler.sample(n=100, burn_in=10)

    print(f"Generated {len(particles)} particles")
    if particles:
        first_particle = particles[0]
        print(f"First particle variables: {first_particle.get_variables()}")
        print(f"First particle values: {first_particle.get_all_values()}")

    print("\n3. Traditional CPT-based network:")

    # Example using CPTs (traditional approach)
    from ..ds import ConditionalProbabilityTable as CPT

    with BayesianNetwork() as bn:
        # Prior for X
        X = CPT(table=[{"X": 0, "value": 0.8}, {"X": 1, "value": 0.2}], outcomes=["X"])

        # Conditional probability for Y given X
        Y = CPT(
            table=[
                {"X": 0, "Y": 0, "value": 0.25},
                {"X": 0, "Y": 1, "value": 0.75},
                {"X": 1, "Y": 0, "value": 0.6},
                {"X": 1, "Y": 1, "value": 0.4},
            ],
            givens=["X"],
            outcomes=["Y"],
        )

        # Add to network
        bn.add_node(X)
        bn.add_edge(Y)

        print(f"Network has {len(bn.cpts)} CPTs")
        for var, cpt in bn.cpts.items():
            print(f"  - {var}: {cpt}")

    print("\nExample completed!")


if __name__ == "__main__":
    run_random_variable_example()
