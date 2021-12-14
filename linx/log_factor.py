"""
Log Factor module
"""
import numpy as np
import pandas as pd

from .errors import ArgumentError
from .factor_one import FactorOne


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
    def __init__(self, df=None, cpt=None):
        if df is not None:
            self.df = df.copy()
        else:
            self.df = cpt.df.copy()

        self.__validate__()
        self.copy_df = self.df.copy()

    def __validate__(self):
        variables = self.get_variables()

        counts = self.df.groupby(variables).count()['value']

        if (counts > 1).sum(axis=0) > 0:
            raise ArgumentError(
                f"Dataframe {self.df} must not have duplicate "
                + "entries with variables."
            )

    def __repr__(self):
        return f"\nLogFactor(\nvariables: {self.get_variables()}" \
            + f", \ndf: \n{self.df})\n"

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.df.columns) - {'value'})

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

        return LogFactor(merged[variables])

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

        return LogFactor(merged[variables])

    def __merged__(self, other):
        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        variables = list(left_vars.union(right_vars.union({'value'})))

        if common:
            merged = self.df.merge(other.df, on=common)
        else:
            left_df = self.df.copy()
            right_df = other.df.copy()
            left_df['cross-join'] = 1
            right_df['cross-join'] = 1
            merged = left_df.merge(right_df, on='cross-join')
        return merged, variables

    def sum(self, var):
        """
        Marginalize a variable in log space.

        log(Σ ɸ(X=x, Y)) = log(ɸ(Y))
            x

        Parameters:
            var: string
                The variable to be summed out.

        Returns: Factor
        """
        other_vars = list(set(self.get_variables()) - {'value', var})
        if not other_vars:
            return FactorOne()

        # Could have a lag of one,
        # Remove rows with NaN
        # Run logaddexp
        # Get the even results
        # Run this recursively until we have one item per group.
        # Could do pairwise
        self.copy_df = self.df.copy()
        self.copy_df['rank'] = self.copy_df.groupby(other_vars)['value'].rank()

        # store the last rows that are odd-numbered
        self.copy_df['max_rank'] = \
            self.copy_df.groupby(other_vars)['rank'].max()

        # evens. __compute_log_sum_exp__ perfectly handles the even case.
        self.copy_df = \
            self.copy_df[self.copy_df['max_rank'] % 2 == 0]

        # If a grouping has odd number of rows, and there is more than 1 row,
        # then we'll need to append this to self.copy_df later after
        # self.copy_df is mutated by __compute_log_sum_exp__. At the end of the
        # first call, we'll have processed all the items except the last row
        # for odd-numbered groupings. Thus we append the last row later. Doing
        # so will give us an even number of rows per grouping, which means we
        # can run __compute_log_sum_exp__ again, to give us a final aggregation
        # for those groupings that had originally a set of rows that had
        # greater than 1.
        to_add = \
            self.copy_df[
                (self.copy_df['max_rank'] % 2 == 1) &
                (self.copy_df['max_rank'] != 1)
            ]

        # Finally, after we run the __compute_log_sum_exp__ twice, we'll append
        # the rows that only had one row. Those don't need to be processed.
        only_one_row = \
            self.copy_df[self.copy_df['max_rank'] == 1]

        while any(self.copy_df['rank'] > 1):
            self.__compute_log_sum_exp__(other_vars)
            # self.copy_df.drop(columns=['value_lag'], inplace=True)

        self.copy_df = pd.concat([
            self.copy_df,
            to_add
        ])

        # Combine odds
        self.__compute_log_sum_exp__(other_vars)

        self.copy_df = pd.concat([
            self.copy_df,
            only_one_row
        ])

        return LogFactor(
            df=self.copy_df.reset_index()[other_vars + ['value']]
        )

    def __compute_log_sum_exp__(self, other_vars):
        shifted = self\
            .copy_df\
            .reset_index()\
            .groupby(other_vars)[['value']].shift(-1)
        lagged = self.copy_df.join(
            shifted.rename(columns=lambda x: x+"_lag")
        )
        self.copy_df['value'] = \
            np.logaddexp(lagged['value'], lagged['value_lag'])
        # TODO: this might be a problem when there are an odd number of
        # rows for a group?
        self.copy_df = self.copy_df.dropna()
        self.copy_df['rank'] = self\
            .copy_df.groupby(other_vars)['value'].rank()
        self.copy_df = self.copy_df[self.copy_df['rank'] % 2 == 1]
