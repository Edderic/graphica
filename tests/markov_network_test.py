import pandas as pd

from ..linx.data import ParquetData
from ..linx.ds import MarkovNetwork, ConditionalProbabilityTable as CPT, \
    Factor, Query
from .conftest import (assert_approx_value_df, clean_tmp, get_tmp_path)


def test_apply_query():
    clean_tmp()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    query = Query(
        outcomes=[{'Y': lambda df: df['Y'] == 1}],
        givens=[{'X': lambda df: df['X'] == 1}]
    )

    markov_network = MarkovNetwork()
    factor = Factor(
        cpt=cpt_1
    )
    markov_network.add_factor(factor)
    markov_network.apply_query(query)

    factors = markov_network.get_factors()

    expected_df = pd.DataFrame(
        [
            {'X': 1, 'Y': 1, 'value': 0.4}
        ]
    )

    assert_approx_value_df(
        factors[0].get_df(),
        expected_df,
    )


def test_get_factors():
    clean_tmp()

    markov_network = MarkovNetwork()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        ParquetData(df_2, storage_folder=get_tmp_path()),
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        ParquetData(df_3, storage_folder=get_tmp_path()),
        outcomes=['X'],
    )

    factor_3 = Factor(cpt=cpt_3)
    markov_network.add_factor(factor_3)

    factors = markov_network.get_factors()

    assert factor_1 in factors
    assert factor_2 in factors
    assert factor_3 in factors

    x_factors = markov_network.get_factors('X')

    assert factor_1 in x_factors
    assert factor_2 in x_factors
    assert factor_3 in x_factors

    a_factors = markov_network.get_factors('A')

    assert factor_1 not in a_factors
    assert factor_2 in a_factors
    assert factor_3 not in a_factors


def test_get_variables():
    markov_network = MarkovNetwork()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        ParquetData(df_2, storage_folder=get_tmp_path()),
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        ParquetData(df_3, storage_folder=get_tmp_path()),
        outcomes=['X'],
    )

    factor_3 = Factor(cpt=cpt_3)
    markov_network.add_factor(factor_3)

    assert set(markov_network.get_variables()) == {'X', 'Y', 'A'}


def test_remove_factors():
    markov_network = MarkovNetwork()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        ParquetData(df_2, storage_folder=get_tmp_path()),
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    markov_network.remove_factor(factor_1)

    factors = markov_network.get_factors()
    assert len(factors) == 1
    assert factor_2 in factors
