"""
Log Factor module
"""
from .errors import ArgumentError


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
            + f", \ndf: \n{self.df})"

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.df.columns) - {'value'})

    def add(self, other):
        """
        Parameters:
            other: LogFactor

        Returns:
            LogFactor
        """

        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        if common:
            merged = self.df.merge(other.df, on=common)
        else:
            left_df = self.df.copy()
            right_df = other.df.copy()
            left_df['cross-join'] = 1
            right_df['cross-join'] = 1
            merged = left_df.merge(right_df, on='cross-join')

        merged['value'] = merged.value_x + merged.value_y

        return LogFactor(
            merged[
                list(left_vars.union(right_vars.union({'value'})))
            ]
        )
