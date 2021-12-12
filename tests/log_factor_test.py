import numpy as np
import pandas as pd
from ..linx.log_factor import LogFactor

from .conftest import assert_approx_value_df


def test_log_factor_add():
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.5)},
        {'x': 0, 'y': 1, 'value': np.log(0.6)},
        {'x': 1, 'y': 0, 'value': np.log(0.8)},
        {'x': 1, 'y': 1, 'value': np.log(0.7)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    df_2 = pd.DataFrame([
        {'x': 0, 'value': np.log(0.5)},
        {'x': 1, 'value': np.log(0.2)},
    ])

    log_factor_2 = LogFactor(df=df_2)

    new_log_factor = log_factor_1.add(log_factor_2)

    expected_df = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.25)},
        {'x': 0, 'y': 1, 'value': np.log(0.3)},
        {'x': 1, 'y': 0, 'value': np.log(0.16)},
        {'x': 1, 'y': 1, 'value': np.log(0.14)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )
