import pytest

import pandas as pd

from ..linx.ds import Factors, Factor
from .conftest import clean_tmp, get_tmp_path
from ..linx.data import ParquetData


def test_subscriptable(two_factors):
    factors = two_factors
    first_factor = factors[0]
    second_factor = factors[1]

    assert first_factor.get_variables() == ['X']
    assert second_factor.get_variables() == ['Y']

    clean_tmp()


def test_looping():
    clean_tmp()

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

    expected_factors = [
        factor_1,
        factor_2
    ]

    factors = Factors([factor_1, factor_2])

    for i, factor in enumerate(factors):
        assert factor == expected_factors[i]


def test_get_variables():
    clean_tmp()

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

    factor_1 = Factor(
        ParquetData(df1, storage_folder=get_tmp_path())
    )
    factor_2 = Factor(
        ParquetData(df2, storage_folder=get_tmp_path())
    )
    factor_3 = Factor(
        ParquetData(df3, storage_folder=get_tmp_path())
    )
    factors = Factors([factor_1, factor_2, factor_3])
    variables = factors.get_variables()

    assert {'X', 'Y', 'A'} == variables


@pytest.mark.f
def test_get_filters():
    clean_tmp()

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

    factor_1 = Factor(
        ParquetData(df1, storage_folder=get_tmp_path())
    )
    factor_2 = Factor(
        ParquetData(df2, storage_folder=get_tmp_path())
    )
    factor_3 = Factor(
        ParquetData(df3, storage_folder=get_tmp_path())
    )
    factors = Factors([factor_1, factor_2, factor_3])

    filters = {
        'A': 0
    }

    filtered = factors.filter(filters)

    assert set(filtered[2].get_variables()) == set(['Y', 'A'])
    assert all(filtered[2].get_df()['A'] == 0)

    clean_tmp()
