import pytest
import numpy as np
import pandas as pd
from ..graphica.log_factor import LogFactor
from ..graphica.data import ParquetData

from .conftest import (assert_approx_value_df, get_tmp_path, clean_tmp)


def test_log_factor_add():
    clean_tmp()

    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.5)},
        {'x': 0, 'y': 1, 'value': np.log(0.6)},
        {'x': 1, 'y': 0, 'value': np.log(0.8)},
        {'x': 1, 'y': 1, 'value': np.log(0.7)},
    ])

    log_factor_1 = LogFactor(
        ParquetData(
            df_1,
            storage_folder=get_tmp_path()
        )
    )

    df_2 = pd.DataFrame([
        {'x': 0, 'value': np.log(0.5)},
        {'x': 1, 'value': np.log(0.2)},
    ])

    log_factor_2 = LogFactor(
        ParquetData(
            df_2,
            storage_folder=get_tmp_path()
        )
    )

    new_log_factor = log_factor_1.add(log_factor_2)

    expected_df = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.25)},
        {'x': 0, 'y': 1, 'value': np.log(0.3)},
        {'x': 1, 'y': 0, 'value': np.log(0.16)},
        {'x': 1, 'y': 1, 'value': np.log(0.14)},
    ])

    assert_approx_value_df(
        new_log_factor.data.read(),
        expected_df,
    )


def test_log_factor_subtract():
    clean_tmp()

    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.5)},
        {'x': 0, 'y': 1, 'value': np.log(0.6)},
        {'x': 1, 'y': 0, 'value': np.log(0.9)},
        {'x': 1, 'y': 1, 'value': np.log(0.3)},
    ])

    log_factor_1 = LogFactor(
        ParquetData(
            df_1,
            storage_folder=get_tmp_path()
        )
    )

    df_2 = pd.DataFrame([
        {'x': 0, 'value': np.log(0.1)},
        {'x': 1, 'value': np.log(0.3)},
    ])

    log_factor_2 = LogFactor(
        ParquetData(
            df_2,
            storage_folder=get_tmp_path()
        )
    )

    new_log_factor = log_factor_1.subtract(log_factor_2)

    expected_df = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(5)},
        {'x': 0, 'y': 1, 'value': np.log(6)},
        {'x': 1, 'y': 0, 'value': np.log(3)},
        {'x': 1, 'y': 1, 'value': np.log(1)},
    ])

    assert_approx_value_df(
        new_log_factor.data.read(),
        expected_df,
    )
