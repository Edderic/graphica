"""
Log Factor module
"""
import numpy as np
import pandas as pd

from .errors import ArgumentError
from .factor_one import FactorOne


def compute_log_sum_exp(other_vars, tmp_df):
    """
    Compute log_sum_exp

    Parameters:
        other_vars: list[str]
        tmp_df: pd.DataFrame
    """
    shifted = tmp_df\
        .groupby(other_vars)[['value']].shift(-1)

    lagged = tmp_df.merge(
        shifted.rename(columns=lambda x: x+"_lag"),
        left_index=True,
        right_index=True
    )

    tmp_df['value'] = \
        np.logaddexp(lagged['value'], lagged['value_lag'])
    # TODO: this might be a problem when there are an odd number of
    # rows for a group?
    tmp_df = tmp_df.dropna()

    return tmp_df


class LogFactor:
    """
    Bayesian inference could involve multiplying many probabilities, which
    could lead to underflow. Really tiny probabilities, when multiplied
    together, could surpass the lowest floating point number that Python could
    represent, which could lead to Python assuming that the value of 0., which
    makes the whole product 0.

    Working in the log space helps us prevent underflow error while still
    letting us represent really tiny probabilities.
    """

    def __init__(self, data=None, cpt=None):
        if data is not None:
            self.data = data
        else:
            self.data = cpt.get_data()

        self.__validate__()
        self.data_class = self.data.__class__

    def __validate__(self):
        variables = self.get_variables()

        df = self.data.read()
        counts = df.groupby(variables).count()['value']

        if (counts > 1).sum(axis=0) > 0:
            raise ArgumentError(
                f"Dataframe {df} must not have duplicate "
                + "entries with variables."
            )

        if any(df['value'] == -np.inf):
            raise ArgumentError(
                "Must not have negative infinity values. df:\n"
                + f"{df}"
            )

        if df.shape[0] == 0:
            raise ArgumentError(
                f"Dataframe is empty. Columns: {df.columns}"
            )

    def __repr__(self):
        return f"\nLogFactor(\nvariables: {self.get_variables()}" \
            + f", \ndf: \n{self.data.read()}\n)\n"

    def filter(self, query):
        """
        Apply filters of a query to this factor.

        Parameters:
            query: Query
                Implements get_filters. Each item is a dict.
                key: string
                    Name of the variable.
                value: callable or string or float or integer
                    We use this for filtering.

        Returns: LogFactor
        """
        df = self.data.read()
        filters = query.get_filters()

        for f in filters:
            key = list(f.keys())[0]
            value = list(f.values())[0]

            if key in self.get_variables():
                if callable(value):
                    df = df[value(df)]
                else:
                    # We assume we're doing an equality
                    df = df[df[key] == value]

        return LogFactor(
            self.data.__class__(
                df, storage_folder=self.data.get_storage_folder()
            )
        )

    def get_data(self):
        """
        Return: Data
        """
        return self.data

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.data.get_columns()) - {'value'})

    def add(self, other):
        """
        Addition in log space. Let x be one factor and y be the "other" factor.
        Performs the following operation:
            log(ɸ(x)) + log(ɸ(y)) = log(ɸ(x) * ɸ(y))

        Parameters:
            other: LogFactor

        Returns:
            LogFactor
        """

        merged, variables = self.__merged__(other)
        merged['value'] = merged.value_x + merged.value_y

        data = self.data.__class__(
            merged[variables],
            storage_folder=self.data.get_storage_folder()
        )

        return LogFactor(
            data
        )

    def subtract(self, other):
        """
        Subtraction in log space. Let x be one factor and y be the "other"
        factor.
        Performs the following operation:
            log(ɸ(x)) - log(ɸ(y)) = log(ɸ(x) / ɸ(y))

        Parameters:
            other: LogFactor

        Returns:
            LogFactor
        """

        merged, variables = self.__merged__(other)
        merged['value'] = merged.value_x - merged.value_y

        return LogFactor(
            self.data.__class__(
                merged[variables],
                storage_folder=self.data.get_storage_folder()
            )
        )

    def __merged__(self, other):
        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(left_vars.intersection(right_vars))

        variables = list(left_vars.union(right_vars.union({'value'})))

        left_df = self.data.read()
        right_df = other.data.read()

        if common:
            merged = left_df.merge(right_df, on=common)
        else:
            left_df['cross-join'] = 1
            right_df['cross-join'] = 1
            merged = left_df.merge(right_df, on='cross-join')

        if merged.shape[0] == 0:
            raise ArgumentError(
                "Tables being merged have nothing in common:"
                + "\ncommon: {common}"
                + "\nLeft:\n{left_df[common]},"
                + "\nRight:\n{right_df[common]}"
            )
        return merged, variables
