import pandas as pd

from .conftest import clean_tmp, get_tmp_path
from ..linx.data import ParquetData
from ..linx.variable_elimination import min_fill_edges
from ..linx.ds import BayesianNetwork, ConditionalProbabilityTable as CPT


def test_min_fill_edges():
    bayesian_network = BayesianNetwork()

    df1 = pd.DataFrame([
        {
            'X': 0, 'value': 0.7,
        },
        {
            'X': 1, 'value': 0.3,
        }
    ])

    df2 = pd.DataFrame([
        {
            'Y': 0, 'value': 0.4,
        },
        {
            'Y': 1, 'value': 0.6,
        }
    ])

    df3 = pd.DataFrame([
        {'X': 0, 'Y': 0, 'Z': 0, 'value': 0.4},
        {'X': 0, 'Y': 1, 'Z': 0, 'value': 0.6},
        {'X': 1, 'Y': 0, 'Z': 0, 'value': 0.9},
        {'X': 1, 'Y': 1, 'Z': 0, 'value': 0.1},
        {'X': 0, 'Y': 0, 'Z': 1, 'value': 0.6},
        {'X': 0, 'Y': 1, 'Z': 1, 'value': 0.4},
        {'X': 1, 'Y': 0, 'Z': 1, 'value': 0.1},
        {'X': 1, 'Y': 1, 'Z': 1, 'value': 0.9},
    ])

    df4 = pd.DataFrame([
        {'Z': 0, 'A': 0, 'value': 0.4},
        {'Z': 0, 'A': 1, 'value': 0.6},
        {'Z': 1, 'A': 0, 'value': 0.9},
        {'Z': 1, 'A': 1, 'value': 0.1},
    ])

    cpt_1 = CPT(
        ParquetData(df1, storage_folder=get_tmp_path()),
        outcomes=['X'],
    )

    cpt_2 = CPT(
        ParquetData(df2, storage_folder=get_tmp_path()),
        outcomes=['Y']
    )

    cpt_3 = CPT(
        ParquetData(df3, storage_folder=get_tmp_path()),
        outcomes=['Z'],
        givens=['X', 'Y']
    )

    cpt_4 = CPT(
        ParquetData(df4, storage_folder=get_tmp_path()),
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

    clean_tmp()
