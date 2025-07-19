"""
Factor module
"""

import numpy as np

from .errors import ArgumentError
from .factor_one import FactorOne


class Factor:
    """
    Factor class.

    A factor is something that can be multiplied with another factor.
    """

    def __init__(self, data=None, cpt=None, log_factor=None):
        if log_factor is not None and data is not None:
            raise ArgumentError(
                "Factor must be supplied with only one of"
                + " Data, ConditionalProbabilityTable, or LogFactor."
            )
        if cpt is not None and data is not None:
            raise ArgumentError(
                "Factor must be supplied with only one of"
                + " Data, ConditionalProbabilityTable, or LogFactor"
            )

        if data is not None:
            self.data = data
        elif cpt is not None:
            self.data = cpt.get_data()
        else:
            log_data = log_factor.get_data()
            df = log_data.read()
            df.loc[:, "value"] = np.exp(df["value"])
            self.data = log_data.__class__(df, log_data.get_storage_folder())

        self.data_class = self.data.__class__

        self.__validate__()

    def __validate__(self):
        df = self.data.read()

        if df.shape[0] == 0:
            raise ArgumentError(f"Dataframe is empty. Columns: {df.columns}")

        cols = list(set(df.columns) - {"value"})

        counts = df.groupby(cols).count()["value"]

        if any(counts > 1):
            raise ArgumentError(f"Too many counts detected: \n{counts}")

    def __merged__(self, other):
        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(left_vars.intersection(right_vars))

        variables = list(left_vars.union(right_vars.union({"value"})))

        left_df = self.data.read()
        right_df = other.data.read()

        if common:
            merged = left_df.merge(right_df, on=common)
        else:
            left_df["cross-join"] = 1
            right_df["cross-join"] = 1
            merged = left_df.merge(right_df, on="cross-join")

        if merged.shape[0] == 0:
            raise ArgumentError(
                "Tables being merged have nothing in common:"
                + "\ncommon: {common}"
                + "\nLeft:\n{left_df[common]},"
                + "\nRight:\n{right_df[common]}"
            )
        return merged, variables

    def __repr__(self):
        return (
            f"\nFactor(\nvariables: {self.get_variables()}"
            + f", \ndf: \n{self.get_df()}\n)"
        )

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.data.get_columns()) - {"value"})

    def prod(self, other):
        """
        Multiplication of one factor with "other" factor.

        Parameters:
            other: Factor

        Returns:
            Factor
        """

        merged, variables = self.__merged__(other)
        merged["value"] = merged.value_x * merged.value_y

        data = self.data.__class__(
            merged[variables], storage_folder=self.data.get_storage_folder()
        )

        return Factor(data)

    def div(self, other):
        """
        Parameters:
            other: Factor

        Returns: Factor
        """

        merged, variables = self.__merged__(other)
        merged["value"] = merged.value_x / merged.value_y

        return Factor(
            self.data.__class__(
                merged[variables], storage_folder=self.data.get_storage_folder()
            )
        )

    def filter(self, filters):
        """
        Apply filters of a query to this factor.

        Parameters:
            filters: dict
                Key is a variable name (string).
                Value is either a callable or non-callable

        Returns: Factor
        """
        if "get_filters" in dir(filters):
            fs = filters.get_filters()
            dictionary = {}
            for d in fs:
                for k, v in d.items():
                    dictionary[k] = v

            fs = dictionary
            # TODO: maybe simpler if get_filters
            # returns a dictionary?
        else:
            fs = filters

        df = self.data.read()

        for key, value in fs.items():
            if key in self.get_variables():
                if callable(value):
                    df = df[value(df)]
                else:
                    # We assume we're doing an equality
                    df = df[df[key] == value]

                if df.shape[0] == 0:
                    raise ArgumentError(
                        "Dataframe is empty after filtering using filter"
                        + f" {key}.\n\tColumns: {df.columns}"
                    )

        return Factor(
            self.data.__class__(df, storage_folder=self.data.get_storage_folder())
        )

    def sum(self, var):
        """
        Get the other variables besides the one being passed in. Group by those
        variables. Take the sum.

        Parameters:
            var: string
                The variable to be summed out.

        Returns: Factor
        """

        variables = var
        if isinstance(var, str):
            variables = [var]

        df = self.get_df()

        other_vars = list(set(self.get_variables()) - set(variables))

        if not other_vars:
            return FactorOne()

        return_df = df.groupby(other_vars).sum()[["value"]].reset_index()

        return Factor(
            data=self.data_class(
                return_df, storage_folder=self.get_data().get_storage_folder()
            )
        )

    def normalize(self, variables=None):
        """
        Make sure the values represent probabilities.

        Parameters:
            variables: list[str]
                The variables in the denominator.

        Returns: Factor
        """

        df = self.get_df()

        if not variables:
            df["value"] = df["value"] / df["value"].sum()

            return Factor(
                data=self.data_class(
                    df, storage_folder=self.get_data().get_storage_folder()
                )
            )

        sum_df = df.groupby(variables)[["value"]].sum()
        merged = df.merge(sum_df, on=variables)
        merged["value"] = merged["value_x"] / merged["value_y"]

        return Factor(
            data=self.data_class(
                merged.drop(columns=["value_x", "value_y"]),
                storage_folder=self.get_data().get_storage_folder(),
            )
        )

    def get_df(self):
        """
        Returns the dataframe.

        Returns: pd.DataFrame
        """

        return self.get_data().read()

    def get_data(self):
        """
        Return the data object.

        Returns: Data
        """
        return self.data
