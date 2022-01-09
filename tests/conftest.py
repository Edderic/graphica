import pandas as pd
import pytest

from ..linx.data import ParquetData
from ..linx.ds import BayesianNetwork, Factors, Factor,\
    ConditionalProbabilityTable as CPT
from ..linx.misc import get_tmp_path, clean_tmp


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
    variables_without_value = list(set(variables) - {'value'})

    actual_sorted = actual_df.sort_values(by=variables_without_value)
    expected_sorted = expected_df[actual_sorted.columns]\
        .sort_values(by=variables_without_value)

    for (_, x), (_, y) in zip(
        actual_sorted.iterrows(),
        expected_sorted.iterrows()
    ):
        for variable in variables:
            assert x[variable] == pytest.approx(
                y[variable],
                abs=abs_tol
            )


def create_binary_prior_cpt(outcome, value_for_1=None):
    if value_for_1 is None:
        value_for_1 = 0.5

    binary_prior_df = pd.DataFrame([
        {outcome: 0, 'value': 1.0 - value_for_1},
        {outcome: 1, 'value': value_for_1},
    ])

    return CPT(
        ParquetData(
            binary_prior_df,
            storage_folder=get_tmp_path()
        ),
        outcomes=[outcome],
    )


def create_binary_CPT(
    given,
    outcome,
    vals
):

    # { x: 1, value: 0.9 }
    # { x: 0, value: 0.8 }

    df_rows = []

    for row in vals:
        df_rows.append({
            given: row[given],
            outcome: 1,
            'value': row['value']
        })

        df_rows.append({
            given: 1 - row[given],
            outcome: 0,
            'value': 1 - row['value']
        })

    df = pd.DataFrame(df_rows)

    return CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=[outcome],
        givens=[given]
    )


def create_prior_df(outcome):
    return pd.DataFrame([
        {outcome: 0, 'value': 0.20},
        {outcome: 1, 'value': 0.20},
        {outcome: 2, 'value': 0.20},
        {outcome: 3, 'value': 0.20},
        {outcome: 4, 'value': 0.20},
    ])


def create_df_easy(given, outcome):
    return pd.DataFrame([
        # If given is a 4, then Outcome will most likely be a 4
        {given: 4, outcome: 4, 'value': 0.99},
        {given: 4, outcome: 3, 'value': 0.01},
        # If given is a 3, then Outcome will still likely be high
        {given: 3, outcome: 4, 'value': 0.90},
        {given: 3, outcome: 3, 'value': 0.10},
        # If given is a 2, then Outcome will still likely be high
        {given: 2, outcome: 4, 'value': 0.10},
        {given: 2, outcome: 3, 'value': 0.80},
        {given: 2, outcome: 2, 'value': 0.10},
        # If Given is a 1, then Outcome will most likely score a
        # 1, but a 1 or 3 are possible.
        {given: 1, outcome: 3, 'value': 0.10},
        {given: 1, outcome: 2, 'value': 0.40},
        {given: 1, outcome: 1, 'value': 0.50},
        # If Given is a 0, then Outcome will most likely score a
        # 0, but a tiny chance a 1 will be scored
        {given: 0, outcome: 1, 'value': 0.01},
        {given: 0, outcome: 0, 'value': 0.99},
    ])


def create_df_medium(given, outcome):
    return pd.DataFrame([
        # If outcome is a 4, then Outcome will most likely score a
        # 4, but a 3 or a 2 is possible, due to measurement error.
        {given: 4, outcome: 4, 'value': 0.9},
        {given: 4, outcome: 3, 'value': 0.07},
        {given: 4, outcome: 2, 'value': 0.03},
        # If Given is a 3, then Outcome will most likely score a
        # 3, but 2 or 4 are possible.
        {given: 3, outcome: 4, 'value': 0.05},
        {given: 3, outcome: 3, 'value': 0.9},
        {given: 3, outcome: 2, 'value': 0.05},
        # If Given is a 2, then Outcome will most likely score a
        # 2, but a 1 or 3 are possible.
        {given: 2, outcome: 3, 'value': 0.05},
        {given: 2, outcome: 2, 'value': 0.9},
        {given: 2, outcome: 1, 'value': 0.05},
        # If Given is a 1, then Outcome will most likely score a
        # 1, but a 1 or 3 are possible.
        {given: 1, outcome: 2, 'value': 0.05},
        {given: 1, outcome: 1, 'value': 0.93},
        {given: 1, outcome: 0, 'value': 0.02},
        # If Given is a 0, then Outcome will most likely score a
        # 0, or possibly a 1
        {given: 0, outcome: 1, 'value': 0.01},
        {given: 0, outcome: 0, 'value': 0.99},
    ])


def create_df_hard(given, outcome):
    return pd.DataFrame([
        # If outcome is a 4, then Outcome will most likely score a
        # 4, but a 3 or a 2 is possible, due to measurement error.
        {given: 4, outcome: 4, 'value': 0.3},
        {given: 4, outcome: 3, 'value': 0.4},
        {given: 4, outcome: 2, 'value': 0.3},
        # If Given is a 3, then Outcome will most likely score a
        # 3, but 2 or 4 are possible.
        {given: 3, outcome: 4, 'value': 0.1},
        {given: 3, outcome: 3, 'value': 0.4},
        {given: 3, outcome: 2, 'value': 0.5},
        # If Given is a 2, then Outcome will most likely score a
        # 2, but a 1 or 3 are possible.
        {given: 2, outcome: 2, 'value': 0.4},
        {given: 2, outcome: 1, 'value': 0.6},
        # If Given is a 1, then Outcome will most likely score a
        # 1, but a 1 or 3 are possible.
        {given: 1, outcome: 2, 'value': 0.05},
        {given: 1, outcome: 1, 'value': 0.93},
        {given: 1, outcome: 0, 'value': 0.02},
        # If Given is a 0, then Outcome will most likely score a
        # 0, or possibly a 1
        {given: 0, outcome: 1, 'value': 0.01},
        {given: 0, outcome: 0, 'value': 0.99},
    ])


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

    factor_1 = Factor(
        ParquetData(df1, storage_folder=get_tmp_path())
    )
    factor_2 = Factor(
        ParquetData(df2, storage_folder=get_tmp_path())
    )

    return Factors([factor_1, factor_2])


@pytest.fixture
def two_vars_unconnected_bn():
    df1 = pd.DataFrame([
        {
            'X': 0, 'value': 0.1,
        },
        {
            'X': 1, 'value': 0.9,
        }
    ])

    df2 = pd.DataFrame([
        {
            'Y': 0, 'value': 0.7,
        },
        {
            'Y': 1, 'value': 0.3,
        }
    ])

    bayesian_network = BayesianNetwork(
        priors=[
            CPT(
                ParquetData(df1, storage_folder=get_tmp_path()),
                outcomes=['X'],
            ),
            CPT(
                ParquetData(df2, storage_folder=get_tmp_path()),
                outcomes=['Y'],
            )
        ]
    )

    return bayesian_network
