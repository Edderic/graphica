import pandas as pd
import pytest

from ..linx.ds import BayesianNetwork, Factors, Factor,\
    ConditionalProbabilityTable as CPT


def assert_approx_value_df(
    actual_df,
    expected_df,
    abs_tol=None
):
    """
    Parameters:
        actual_df: pd.DataFrame

        expected_df: pd.DataFrame

        abs_tol: float. Defaults to 0.01.
            Absolute tolerance for approximation.

    Returns: boolean
    """
    if abs_tol is None:
        abs_tol = 0.01

    variables = list(expected_df.columns)
    actual_sorted = actual_df.sort_values(by=variables)
    expected_sorted = expected_df.sort_values(by=variables)

    for (_, x), (_, y) in zip(
        actual_sorted.iterrows(),
        expected_sorted.iterrows()
    ):
        assert x['value'] == pytest.approx(
            y['value'],
            abs=abs_tol
        )

        for variable in variables:
            assert x[variable] == pytest.approx(
                y[variable],
                abs=abs_tol
            )

        assert x['Y'] == pytest.approx(
            y['Y'],
            abs=abs_tol
        )


@pytest.fixture
def collider_and_descendant():
    r"""
        X      Y
         \    /
          v  v
           Z
           |
           v
           A
    """
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

    return bayesian_network


@pytest.fixture
def two_factors():
    df1 = pd.DataFrame([
        {
            'X': 0, 'value': 123,
        },
        {
            'X': 1, 'value': 123,
        }
    ])

    df2 = pd.DataFrame([
        {
            'Y': 0, 'value': 123,
        },
        {
            'Y': 1, 'value': 123,
        }
    ])

    factor_1 = Factor(df=df1)
    factor_2 = Factor(df=df2)

    return Factors([factor_1, factor_2])
