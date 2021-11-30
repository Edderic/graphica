import pandas as pd

from ..linx.ds import MarkovNetwork, ConditionalProbabilityTable as CPT, \
    Factor


def test_get_factors():
    markov_network = MarkovNetwork()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'count': 0.4},
        {'X': 0, 'A': 1, 'count': 0.6},
        {'X': 1, 'A': 0, 'count': 0.7},
        {'X': 1, 'A': 1, 'count': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'count': 0.2},
        {'X': 1, 'count': 0.8}
    ])

    cpt_3 = CPT(
        df=df_3,
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
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'count': 0.4},
        {'X': 0, 'A': 1, 'count': 0.6},
        {'X': 1, 'A': 0, 'count': 0.7},
        {'X': 1, 'A': 1, 'count': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'count': 0.2},
        {'X': 1, 'count': 0.8}
    ])

    cpt_3 = CPT(
        df=df_3,
        outcomes=['X'],
    )

    factor_3 = Factor(cpt=cpt_3)
    markov_network.add_factor(factor_3)

    assert set(markov_network.get_variables()) == {'X', 'Y', 'A'}


def test_remove_factors():
    markov_network = MarkovNetwork()

    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        givens=['X']
    )

    factor_1 = Factor(cpt=cpt_1)
    markov_network.add_factor(factor_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'count': 0.4},
        {'X': 0, 'A': 1, 'count': 0.6},
        {'X': 1, 'A': 0, 'count': 0.7},
        {'X': 1, 'A': 1, 'count': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        givens=['X']
    )

    factor_2 = Factor(cpt=cpt_2)
    markov_network.add_factor(factor_2)

    markov_network.remove_factor(factor_1)

    factors = markov_network.get_factors()
    assert len(factors) == 1
    assert factor_2 in factors
