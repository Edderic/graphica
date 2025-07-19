"""
Bayesian to Markov Network Conversion Example

This example demonstrates how to convert a Bayesian network to a Markov network.
"""

from graphica.bayesian_network import BayesianNetwork
from graphica.conditional_probability_table import ConditionalProbabilityTable as CPT
from graphica.query import Query


def main():
    """Demonstrate Bayesian Network to Markov Network conversion."""

    # Create a simple Bayesian Network: X -> Y -> Z
    bn = BayesianNetwork()

    # Prior for X
    cpt_x = CPT(
        table=[{"X": 0, "value": 0.3}, {"X": 1, "value": 0.7}], outcomes=["X"], name="X"
    )
    bn.add_node(cpt_x)

    # Y depends on X
    cpt_y = CPT(
        table=[
            {"X": 0, "Y": 0, "value": 0.8},
            {"X": 0, "Y": 1, "value": 0.2},
            {"X": 1, "Y": 0, "value": 0.4},
            {"X": 1, "Y": 1, "value": 0.6},
        ],
        outcomes=["Y"],
        givens=["X"],
        name="Y",
    )
    bn.add_node(cpt_y)

    # Z depends on Y
    cpt_z = CPT(
        table=[
            {"Y": 0, "Z": 0, "value": 0.9},
            {"Y": 0, "Z": 1, "value": 0.1},
            {"Y": 1, "Z": 0, "value": 0.2},
            {"Y": 1, "Z": 1, "value": 0.8},
        ],
        outcomes=["Z"],
        givens=["Y"],
        name="Z",
    )
    bn.add_node(cpt_z)

    print("Bayesian Network created:")
    print(f"Nodes: {list(bn.get_random_variables().keys())}")
    print(f"Graph structure: {bn.get_nodes()}")
    print("Edges: X->Y, Y->Z")
    print()

    # Sample from the Bayesian Network
    print("Sampling from Bayesian Network:")
    for i in range(3):
        particle = bn.sample()
        print(
            f"Sample {i+1}: X={particle.get_value('X')}, "
            f"Y={particle.get_value('Y')}, Z={particle.get_value('Z')}"
        )
    print()

    # Convert to Markov Network
    print("Converting to Markov Network...")
    mn = bn.to_markov_network()

    print("Markov Network created:")
    print(f"Variables: {mn.get_variables()}")
    print(f"Number of factors: {len(mn.get_factors())}")
    print()

    # Show the factors in the Markov Network
    print("Factors in Markov Network:")
    for i, factor in enumerate(mn.get_factors()):
        print(f"Factor {i+1}:")
        print(f"  Variables: {factor.get_variables()}")
        print(f"  Data:\n{factor.get_df()}")
        print()

    # Demonstrate that the Markov Network can be used for inference
    print("Using Markov Network for inference:")

    # Create a query: P(X=1 | Z=1)
    query = Query(outcomes=["X"], givens=[{"Z": 1}])

    # Apply the query to the Markov Network
    mn.apply_query(query)

    print("After applying query P(X=1 | Z=1):")
    for i, factor in enumerate(mn.get_factors()):
        print(f"Factor {i+1} after filtering:")
        print(f"  Variables: {factor.get_variables()}")
        print(f"  Data:\n{factor.get_df()}")
        print()

    print("Conversion successful! The Bayesian Network has been converted")
    print("to a Markov Network where each CPT became a factor.")


if __name__ == "__main__":
    main()
