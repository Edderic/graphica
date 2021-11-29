import pandas as pd
import pytest
from ..linx.ds import ConditionalProbabilityTable as CPT, Factor


def test_factor_div():
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        given=['X']
    )

    factor_1 = Factor(cpt=cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'count': 0.4},
        {'X': 0, 'A': 1, 'count': 0.6},
        {'X': 1, 'A': 0, 'count': 0.7},
        {'X': 1, 'A': 1, 'count': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        given=['X']
    )

    factor_2 = Factor(cpt=cpt_2)

    factor_3 = factor_1.div(factor_2)

    expected_factor_df = pd.DataFrame([
        {'A': 0, 'X': 0, 'Y': 0, 'count': 0.625},
        {'A': 1, 'X': 0, 'Y': 0, 'count': 0.416},
        {'A': 0, 'X': 0, 'Y': 1, 'count': 1.875},
        {'A': 1, 'X': 0, 'Y': 1, 'count': 1.25},
        {'A': 0, 'X': 1, 'Y': 0, 'count': 0.857},
        {'A': 1, 'X': 1, 'Y': 0, 'count': 2.0},
        {'A': 0, 'X': 1, 'Y': 1, 'count': 0.571},
        {'A': 1, 'X': 1, 'Y': 1, 'count': 1.33},
    ])
    indexed_left = factor_3.df.set_index(["A", "X", "Y"])
    indexed_right = expected_factor_df.set_index(["A", "X", "Y"])

    for (_, left), (_, right) in zip(
        indexed_left.iterrows(), indexed_right.iterrows()
    ):
        assert left['count'] == pytest.approx(right['count'], abs=0.01)


def test_factor_prod():
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        given=['X']
    )

    factor_1 = Factor(cpt=cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'count': 0.4},
        {'X': 0, 'A': 1, 'count': 0.6},
        {'X': 1, 'A': 0, 'count': 0.7},
        {'X': 1, 'A': 1, 'count': 0.3},
    ])

    cpt_2 = CPT(
        df=df_2,
        outcomes=['A'],
        given=['X']
    )

    factor_2 = Factor(cpt=cpt_2)

    factor_3 = factor_1.prod(factor_2)

    expected_factor_df = pd.DataFrame([
        {'A': 0, 'X': 0, 'Y': 0, 'count': 0.1},
        {'A': 1, 'X': 0, 'Y': 0, 'count': 0.15},
        {'A': 0, 'X': 0, 'Y': 1, 'count': 0.3},
        {'A': 1, 'X': 0, 'Y': 1, 'count': 0.45},
        {'A': 0, 'X': 1, 'Y': 0, 'count': 0.42},
        {'A': 1, 'X': 1, 'Y': 0, 'count': 0.18},
        {'A': 0, 'X': 1, 'Y': 1, 'count': 0.28},
        {'A': 1, 'X': 1, 'Y': 1, 'count': 0.12},
    ])

    indexed_left = factor_3.df.set_index(["A", "X", "Y"])
    indexed_right = expected_factor_df.set_index(["A", "X", "Y"])

    for (_, left), (_, right) in zip(
        indexed_left.iterrows(), indexed_right.iterrows()
    ):
        assert left['count'] == pytest.approx(right['count'])


def test_factor_sum():
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'count': 0.25},
        {'X': 0, 'Y': 1, 'count': 0.75},
        {'X': 1, 'Y': 0, 'count': 0.6},
        {'X': 1, 'Y': 1, 'count': 0.4},
    ])

    cpt_1 = CPT(
        df=df,
        outcomes=['Y'],
        given=['X']
    )

    factor = Factor(cpt=cpt_1)

    new_factor = factor.sum('Y')

    expected_df = pd.DataFrame([
        {
            'X': 0, 'count': 1.0
        },
        {
            'X': 1, 'count': 1.0
        }
    ])

    assert new_factor.df.equals(expected_df)
