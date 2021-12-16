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


def test_log_factor_subtract():
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.5)},
        {'x': 0, 'y': 1, 'value': np.log(0.6)},
        {'x': 1, 'y': 0, 'value': np.log(0.9)},
        {'x': 1, 'y': 1, 'value': np.log(0.3)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    df_2 = pd.DataFrame([
        {'x': 0, 'value': np.log(0.1)},
        {'x': 1, 'value': np.log(0.3)},
    ])

    log_factor_2 = LogFactor(df=df_2)

    new_log_factor = log_factor_1.subtract(log_factor_2)

    expected_df = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(5)},
        {'x': 0, 'y': 1, 'value': np.log(6)},
        {'x': 1, 'y': 0, 'value': np.log(3)},
        {'x': 1, 'y': 1, 'value': np.log(1)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )


def test_sum_even_groupings_1():
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.3)},
        {'x': 0, 'y': 1, 'value': np.log(0.1)},
        {'x': 1, 'y': 0, 'value': np.log(0.7)},
        {'x': 1, 'y': 1, 'value': np.log(0.9)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    new_log_factor = log_factor_1.sum('x')

    expected_df = pd.DataFrame([
        {'y': 0, 'value': np.log(1)},
        {'y': 1, 'value': np.log(1)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )


def test_sum_even_groupings_2():
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.1)},
        {'x': 0, 'y': 1, 'value': np.log(0.2)},
        {'x': 1, 'y': 0, 'value': np.log(0.3)},
        {'x': 1, 'y': 1, 'value': np.log(0.4)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    new_log_factor = log_factor_1.sum('x')

    expected_df = pd.DataFrame([
        {'y': 0, 'value': np.log(0.4)},
        {'y': 1, 'value': np.log(0.6)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )


def test_sum_odd_groupings_1():
    """
    When there are odd number of groupings, then they should be
    processed accordingly.
    """
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 0, 'value': np.log(0.1)},
        {'x': 0, 'y': 1, 'value': np.log(0.2)},
        {'x': 0, 'y': 2, 'value': np.log(0.2)},
        {'x': 1, 'y': 0, 'value': np.log(0.3)},
        {'x': 1, 'y': 1, 'value': np.log(0.4)},
        {'x': 1, 'y': 2, 'value': np.log(0.1)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    new_log_factor = log_factor_1.sum('y')

    expected_df = pd.DataFrame([
        {'x': 0, 'value': np.log(0.5)},
        {'x': 1, 'value': np.log(0.8)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )


def test_sum_even_and_odd_groupings_1():
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 1, 'value': np.log(0.2)},
        {'x': 0, 'y': 2, 'value': np.log(0.2)},
        {'x': 1, 'y': 0, 'value': np.log(0.3)},
        {'x': 1, 'y': 1, 'value': np.log(0.4)},
        {'x': 1, 'y': 2, 'value': np.log(0.1)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    new_log_factor = log_factor_1.sum('y')

    expected_df = pd.DataFrame([
        {'x': 0, 'value': np.log(0.4)},
        {'x': 1, 'value': np.log(0.8)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )


def test_sum_only_one_row_groupings_1():
    """
    It should do nothing to the data.
    """
    df_1 = pd.DataFrame([
        {'x': 0, 'y': 1, 'value': np.log(0.2)},
        {'x': 1, 'y': 2, 'value': np.log(0.1)},
    ])

    log_factor_1 = LogFactor(df=df_1)

    new_log_factor = log_factor_1.sum('y')

    expected_df = pd.DataFrame([
        {'x': 0, 'value': np.log(0.2)},
        {'x': 1, 'value': np.log(0.1)},
    ])

    assert_approx_value_df(
        new_log_factor.df,
        expected_df,
    )
