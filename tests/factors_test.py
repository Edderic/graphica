import pandas as pd

from ..linx.ds import Factors, Factor


def test_looping():
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

    expected_factors = [
        factor_1,
        factor_2
    ]

    factors = Factors([factor_1, factor_2])

    for i, factor in enumerate(factors):
        assert factor == expected_factors[i]


def test_get_variables():
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

    df3 = pd.DataFrame([
        {
            'Y': 0, 'A': 0, 'value': 123,
        },
        {
            'Y': 0, 'A': 1, 'value': 123,
        },
        {
            'Y': 1, 'A': 0, 'value': 123,
        },
        {
            'Y': 1, 'A': 1, 'value': 123,
        },
    ])

    factor_1 = Factor(df=df1)
    factor_2 = Factor(df=df2)
    factor_3 = Factor(df=df3)
    factors = Factors([factor_1, factor_2, factor_3])
    variables = factors.get_variables()

    assert {'X', 'Y', 'A'} == variables
