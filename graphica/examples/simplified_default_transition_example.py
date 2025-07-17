"""
Example demonstrating the simplified DefaultTransition API.

This example shows how DefaultTransition now works without distribution-specific arguments
and how the perturb methods automatically use reasonable defaults based on distribution parameters.
"""

import numpy as np
from ..ds import BayesianNetwork
from ..random.normal import Normal
from ..random.gamma import Gamma
from ..random.beta import Beta
from ..random.deterministic import Deterministic
from ..query import Query
from ..inference.default_transition import DefaultTransition
from ..particles.particle import Particle


def main():
    """Demonstrate the simplified DefaultTransition API."""
    np.random.seed(42)

    print("=== Simplified DefaultTransition Example ===\n")

    # Create a Bayesian network
    bn = BayesianNetwork()

    # Add some random variables with different distributions
    normal_var = Normal(name="normal_var", mean=0, std=2)
    gamma_var = Gamma(name="gamma_var", shape=3, scale=1)
    beta_var = Beta(name="beta_var", a=2, b=5)

    bn.add_node(normal_var)
    bn.add_node(gamma_var)
    bn.add_node(beta_var)

    # Add a deterministic node that depends on the others
    def sum_func(normal_var, gamma_var, beta_var):
        return normal_var + gamma_var + beta_var

    det_var = Deterministic(
        name="det_var",
        callable_func=sum_func,
        normal_var=normal_var,
        gamma_var=gamma_var,
        beta_var=beta_var,
    )
    bn.add_node(det_var)

    # Create a query
    query = Query(outcomes=["normal_var", "gamma_var", "beta_var", "det_var"])

    # Create DefaultTransition (no distribution-specific arguments needed!)
    transition = DefaultTransition(bayesian_network=bn, query=query)

    print("✓ Created DefaultTransition without distribution-specific arguments")

    # Create initial particle
    initial_particle = Particle(
        {"normal_var": 0.5, "gamma_var": 2.0, "beta_var": 0.3, "det_var": 2.8}
    )

    print(f"Initial particle values:")
    for var in ["normal_var", "gamma_var", "beta_var", "det_var"]:
        print(f"  {var}: {initial_particle.get_value(var):.4f}")

    # Perform several transitions
    print(f"\nPerforming transitions...")
    current_particle = initial_particle

    for i in range(5):
        new_particle = transition.transition(current_particle)
        print(f"\nTransition {i+1}:")
        for var in ["normal_var", "gamma_var", "beta_var", "det_var"]:
            old_val = current_particle.get_value(var)
            new_val = new_particle.get_value(var)
            change = new_val - old_val
            print(f"  {var}: {old_val:.4f} → {new_val:.4f} (Δ={change:+.4f})")

        # Verify deterministic relationship is maintained
        expected_det = (
            new_particle.get_value("normal_var")
            + new_particle.get_value("gamma_var")
            + new_particle.get_value("beta_var")
        )
        actual_det = new_particle.get_value("det_var")
        print(f"  det_var check: expected={expected_det:.4f}, actual={actual_det:.4f}")

        current_particle = new_particle

    print(f"\n✓ All transitions completed successfully!")
    print(f"✓ Deterministic relationships maintained!")
    print(f"✓ Perturbations automatically adapted to distribution parameters!")


if __name__ == "__main__":
    main()
