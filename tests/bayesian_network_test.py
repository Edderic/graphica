import pandas as pd

from ..linx.ds import BayesianNetwork as BN, ConditionalProbabilityTable as CPT


def test_find_cpt_for_node():
    bayesian_network = BN()
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        df=df_3,
        outcomes=['X'],
    )

    bayesian_network.add_node(cpt_3)

    assert bayesian_network.find_cpt_for_node('X') == cpt_3
    assert bayesian_network.find_cpt_for_node('A') == cpt_2
    assert bayesian_network.find_cpt_for_node('Y') == cpt_1


def test_to_markov_network():
    bayesian_network = BN()
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        df=df_3,
        outcomes=['X'],
    )

    bayesian_network.add_node(cpt_3)

    markov_network = bayesian_network.to_markov_network()
    factors = markov_network.get_factors()
    assert len(factors) == 3
