import pandas as pd
import pytest
from ..linx.ds import ConditionalProbabilityTable as CPT, Factor
from ..linx.errors import ArgumentError
from ..linx.query import Query

from .conftest import assert_approx_value_df


def test_duplicate_entry_for_variables():
    """
    There's already an entry for X:0, Y:0. This is not valid. It should raise
    an error.
    """
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 0, 'value': 0.99},
    ])

    with pytest.raises(ArgumentError):
        Factor(df=df)


def test_factor_div():
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

    factor_1 = Factor(cpt=cpt_1)

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

    factor_2 = Factor(cpt=cpt_2)

    factor_3 = factor_1.div(factor_2)

    expected_factor_df = pd.DataFrame([
        {'A': 0, 'X': 0, 'Y': 0, 'value': 0.625},
        {'A': 1, 'X': 0, 'Y': 0, 'value': 0.416},
        {'A': 0, 'X': 0, 'Y': 1, 'value': 1.875},
        {'A': 1, 'X': 0, 'Y': 1, 'value': 1.25},
        {'A': 0, 'X': 1, 'Y': 0, 'value': 0.857},
        {'A': 1, 'X': 1, 'Y': 0, 'value': 2.0},
        {'A': 0, 'X': 1, 'Y': 1, 'value': 0.571},
        {'A': 1, 'X': 1, 'Y': 1, 'value': 1.33},
    ])
    indexed_left = factor_3.get_df().set_index(["A", "X", "Y"])
    indexed_right = expected_factor_df.set_index(["A", "X", "Y"])

    for (_, left), (_, right) in zip(
        indexed_left.iterrows(), indexed_right.iterrows()
    ):
        assert left['value'] == pytest.approx(right['value'], abs=0.01)


def test_factor_prod():
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

    factor_1 = Factor(cpt=cpt_1)

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

    factor_2 = Factor(cpt=cpt_2)

    factor_3 = factor_1.prod(factor_2)

    expected_factor_df = pd.DataFrame([
        {'A': 0, 'X': 0, 'Y': 0, 'value': 0.1},
        {'A': 1, 'X': 0, 'Y': 0, 'value': 0.15},
        {'A': 0, 'X': 0, 'Y': 1, 'value': 0.3},
        {'A': 1, 'X': 0, 'Y': 1, 'value': 0.45},
        {'A': 0, 'X': 1, 'Y': 0, 'value': 0.42},
        {'A': 1, 'X': 1, 'Y': 0, 'value': 0.18},
        {'A': 0, 'X': 1, 'Y': 1, 'value': 0.28},
        {'A': 1, 'X': 1, 'Y': 1, 'value': 0.12},
    ])

    indexed_left = factor_3.get_df().set_index(["A", "X", "Y"])
    indexed_right = expected_factor_df.set_index(["A", "X", "Y"])

    for (_, left), (_, right) in zip(
        indexed_left.iterrows(), indexed_right.iterrows()
    ):
        assert left['value'] == pytest.approx(right['value'])


def test_factor_sum():
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

    factor = Factor(cpt=cpt_1)

    new_factor = factor.sum('Y')

    expected_df = pd.DataFrame([
        {
            'X': 0, 'value': 1.0
        },
        {
            'X': 1, 'value': 1.0
        }
    ])

    assert new_factor.get_df().equals(expected_df)


def test_factor_filter_string():
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

    factor = Factor(cpt=cpt_1)

    query = Query(
        outcomes=[{'Y': 1}],
        givens=[{'X': 1}]
    )
    new_factor = factor.filter(query)

    expected_df = pd.DataFrame(
        [
            {'X': 1, 'Y': 1, 'value': 0.4}
        ]
    )

    new_factor.get_df().reset_index()

    assert_approx_value_df(
        new_factor.get_df(),
        expected_df,
    )


def test_factor_filter():
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

    factor = Factor(cpt=cpt_1)

    query = Query(
        outcomes=[{'Y': lambda df: df['Y'] == 1}],
        givens=[{'X': lambda df: df['X'] == 1}]
    )
    new_factor = factor.filter(query)

    expected_df = pd.DataFrame(
        [
            {'X': 1, 'Y': 1, 'value': 0.4}
        ]
    )

    new_factor.get_df().reset_index()

    assert_approx_value_df(
        new_factor.get_df(),
        expected_df,
    )
