"""
Factor class
"""
from .errors import ArgumentError


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
        return f"Factor({self.get_variables()})"

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
                Has get_filters()

        Returns: Factor
        """
        df = self.df
        filters = query.get_filters()

        for f in filters:
            key = list(f.keys())[0]
            func = list(f.values())[0]

            if key in self.get_variables():
                df = df[func(df)]

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

        merged = self.df.merge(other.df, on=common)
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
        new_df = self.df.groupby(other_vars)\
            .sum()[['value']].reset_index()

        return Factor(new_df)
