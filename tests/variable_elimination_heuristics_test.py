import pandas as pd

from ..linx.infer import min_fill_edges
from ..linx.ds import BayesianNetwork, ConditionalProbabilityTable as CPT


def test_min_fill_edges():
    bayesian_network = BayesianNetwork()

    df1 = pd.DataFrame([
        {
            'X': 0, 'count': 0.7,
        },
        {
            'X': 1, 'count': 0.3,
        }
    ])

    df2 = pd.DataFrame([
        {
            'Y': 0, 'count': 0.4,
        },
        {
            'Y': 1, 'count': 0.6,
        }
    ])

    df3 = pd.DataFrame([
        {'X': 0, 'Y': 0, 'Z': 0, 'count': 0.4},
        {'X': 0, 'Y': 1, 'Z': 0, 'count': 0.6},
        {'X': 1, 'Y': 0, 'Z': 0, 'count': 0.9},
        {'X': 1, 'Y': 1, 'Z': 0, 'count': 0.1},
        {'X': 0, 'Y': 0, 'Z': 1, 'count': 0.6},
        {'X': 0, 'Y': 1, 'Z': 1, 'count': 0.4},
        {'X': 1, 'Y': 0, 'Z': 1, 'count': 0.1},
        {'X': 1, 'Y': 1, 'Z': 1, 'count': 0.9},
    ])

    df4 = pd.DataFrame([
        {'Z': 0, 'A': 0, 'count': 0.4},
        {'Z': 0, 'A': 1, 'count': 0.6},
        {'Z': 1, 'A': 0, 'count': 0.9},
        {'Z': 1, 'A': 1, 'count': 0.1},
    ])

    cpt_1 = CPT(
        df=df1,
        outcomes=['X'],
    )

    cpt_2 = CPT(
        df=df2,
        outcomes=['Y']
    )

    cpt_3 = CPT(
        df=df3,
        outcomes=['Z'],
        givens=['X', 'Y']
    )

    cpt_4 = CPT(
        df=df4,
        outcomes=['A'],
        givens=['Z']
    )

    for cpt in [cpt_1, cpt_2, cpt_3, cpt_4]:
        bayesian_network.add_edge(
            cpt=cpt
        )

    best_choice, min_number_of_vars = min_fill_edges(
        eliminateables=['X', 'Y', 'Z', 'A'],
        network=bayesian_network.to_markov_network()
    )

    assert best_choice == 'A'
    assert min_number_of_vars == 2
