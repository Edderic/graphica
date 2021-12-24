"""
Log Factor module
"""
import numpy as np
import pandas as pd

from .data import ParquetData
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
                "Dataframe is empty."
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
            ParquetData(
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
        return list(set(self.data.read().columns) - {'value'})

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

        return LogFactor(
            ParquetData(
                merged[variables],
                storage_folder=self.data.get_storage_folder()
            )
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
            ParquetData(
                merged[variables],
                storage_folder=self.data.get_storage_folder()
            )
        )

    def __merged__(self, other):
        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        variables = list(left_vars.union(right_vars.union({'value'})))

        df = self.data.read()

        if common:
            merged = df.merge(other.data.read(), on=common)
        else:
            left_df = df
            right_df = other.data.read()
            left_df['cross-join'] = 1
            right_df['cross-join'] = 1
            merged = left_df.merge(right_df, on='cross-join')
        return merged, variables

    def sum(self, variables):
        """
        Marginalize a variable in log space.

        log(Σ ɸ(X=x, Y)) = log(ɸ(Y))
            x

        Parameters:
            variables: list or str
                The variable to be summed out.

        Returns: Factor
        """
        if isinstance(variables, str):
            variables = [variables]

        other_vars = list(
            (set(self.get_variables()) - {'value'}) - set(variables)
        )
        if not other_vars:
            return FactorOne()

        # Could have a lag of one,
        # Remove rows with NaN
        # Run logaddexp
        # Get the even results
        # Run this recursively until we have one item per group.
        # Could do pairwise
        tmp_df = self.data.read()
        tmp_df['cumcount'] = tmp_df\
            .groupby(other_vars)['value'].transform('cumcount') + 1

        # store the last rows that are odd-numbered
        tmp_df.loc[:, 'max_cumcount'] = \
            tmp_df.groupby(other_vars)['cumcount'].transform(max)

        # If a grouping has odd number of rows, and there is more than 1 row,
        # then we'll need to append this to tmp_df later after
        # tmp_df is mutated by compute_log_sum_exp. At the end of the
        # first call, we'll have processed all the items except the last row
        # for odd-numbered groupings. Thus we append the last row later. Doing
        # so will give us an even number of rows per grouping, which means we
        # can run compute_log_sum_exp again, to give us a final aggregation
        # for those groupings that had originally a set of rows that had
        # greater than 1.
        only_one_row = \
            tmp_df[
                (tmp_df['max_cumcount'] % 2 == 1) &
                (tmp_df['max_cumcount'] != 1) &
                (tmp_df['cumcount'] == tmp_df['max_cumcount'])
            ]

        # Finally, after we run the compute_log_sum_exp twice, we'll append
        # the rows that only had one row. Those don't need to be processed.
        only_one_row = only_one_row.append(tmp_df[tmp_df['max_cumcount'] == 1])

        # evens. compute_log_sum_exp perfectly handles the even case.
        tmp_df = \
            tmp_df[
                (tmp_df['max_cumcount'] % 2 == 0) |
                (
                    (tmp_df['max_cumcount'] % 2 == 1) &
                    (tmp_df['max_cumcount'] != tmp_df['cumcount'])
                )
            ]

        # do evens
        while any(tmp_df['cumcount'] > 1):
            tmp_df = compute_log_sum_exp(other_vars, tmp_df)
            # See if there's anything to join with the only_one_row
            # get evens
            tmp_df = tmp_df[tmp_df['cumcount'] % 2 == 1]

            tmp_df['cumcount'] = \
                tmp_df.groupby(other_vars)['value'].transform('cumcount') + 1
            tmp_df.loc[:, 'max_cumcount'] = \
                tmp_df.groupby(other_vars)['cumcount'].transform(max)

            # update only_one_row
            only_one_row = only_one_row.append(
                tmp_df[tmp_df['max_cumcount'] == 1]
            )
            tmp_df = tmp_df[tmp_df['max_cumcount'] > 1]

        only_one_row = only_one_row.append(tmp_df)

        only_one_row['cumcount'] = only_one_row.groupby(other_vars)['value']\
            .transform('cumcount') + 1
        only_one_row.loc[:, 'max_cumcount'] = \
            only_one_row.groupby(other_vars)['cumcount'].transform(max)

        even_rows = only_one_row[only_one_row['max_cumcount'] != 1]
        only_one_row = only_one_row[only_one_row['max_cumcount'] == 1]

        tmp_df = even_rows

        tmp_df = compute_log_sum_exp(other_vars, tmp_df)

        vars_to_include = list(set(other_vars).union({'value'}))

        returnables = pd.concat([
            only_one_row[vars_to_include],
            tmp_df[vars_to_include],
        ])

        return LogFactor(
            ParquetData(
                data=returnables.reset_index()[other_vars + ['value']],
                storage_folder=self.data.get_storage_folder()
            )
        )
