"""
Factor class
"""
from .errors import ArgumentError
from .factor_one import FactorOne


class Factor:
    """
    Class for representing factors.
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
        return f"\nFactor(\nvariables: {self.get_variables()}" \
            + f", \ndf: \n{self.df})"

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.df.columns) - {'value'})

    def div(self, other):
        """
        Parameters:
            other: Factor

        Returns: Factor
        """

        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        merged = self.df.merge(other.df, on=common)
        merged['value'] = merged.value_x / merged.value_y

        return Factor(
            merged[
                list(left_vars.union(right_vars.union({'value'})))
            ]
        )

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

        Returns: Factor
        """
        df = self.df
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

        return Factor(df)

    def prod(self, other):
        """
        Parameters:
            other: Factor

        Returns: Factor
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

        merged['value'] = merged.value_x * merged.value_y

        return Factor(
            merged[
                list(left_vars.union(right_vars.union({'value'})))
            ]
        )

    def sum(self, var):
        """
        Parameters:
            var: string
                The variable to be summed out.

        Returns: Factor
        """
        other_vars = list(set(self.get_variables()) - {'value', var})
        if not other_vars:
            return FactorOne()

        new_df = self.df.groupby(other_vars)\
            .sum()[['value']].reset_index()

        return Factor(new_df)

    def normalize(self, variables):
        """
        Make sure the values represent probabilities.

        Parameters:
            variables: list[str]
                The variables in the denominator.

        Returns: Factor
        """
        normalizing_factor = Factor(
            df=self.df
            .groupby(variables)[['value']]
            .sum([['value']])
            .reset_index()
        )

        return self.div(normalizing_factor)
