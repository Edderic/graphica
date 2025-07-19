"""
LogFactorAdapter module
"""

import numpy as np

from .errors import ArgumentError
from .factor_one import FactorOne
from .factor import Factor
from .log_factor import LogFactor


class LogFactorAdapter:
    """
    Acts like a Factor object, but is using a LogFactor underneath.
    """

    def __init__(self, data=None, cpt=None, log_factor=None):
        if log_factor is not None and data is not None:
            raise ArgumentError(
                "LogFactorAdapter must be supplied with only one of"
                + " Data, ConditionalProbabilityTable, or LogFactor."
            )
        if cpt is not None and data is not None:
            raise ArgumentError(
                "LogFactorAdapter must be supplied with only one of"
                + " Data, ConditionalProbabilityTable, or LogFactor"
            )

        self.data = data
        if data is not None:
            df = data.read()
            df["value"] = np.log(df["value"])

            self.log_factor = LogFactor(
                data=data.__class__(df, data.get_storage_folder())
            )
            self.data_class = data.__class__

        elif cpt is not None:
            data = cpt.data

            df = data.read()
            df["value"] = np.log(df["value"])

            self.log_factor = LogFactor(
                data=data.__class__(df, data.get_storage_folder())
            )
            self.data_class = data.__class__

        else:
            self.log_factor = log_factor
            self.data_class = self.log_factor.get_data().__class__

    def __repr__(self):
        return (
            f"\nLogFactorAdapter(\nvariables: {self.get_variables()}"
            + f", \nlog_factor: \n{self.log_factor}\n)"
        )

    def get_variables(self):
        """
        Return variables
        """
        return self.log_factor.get_variables()

    def div(self, other):
        """
        Parameters:
            other: LogFactorAdapter

        Returns: LogFactorAdapter
        """

        return LogFactorAdapter(
            log_factor=self.log_factor.subtract(other.log_factor),
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

        Returns: LogFactorAdapter
        """
        return LogFactorAdapter(log_factor=self.log_factor.filter(query))

    def prod(self, other):
        """
        Parameters:
            other: LogFactorAdapter

        Returns: LogFactorAdapter
        """

        summation = self.log_factor.add(other.log_factor)
        factor = LogFactorAdapter(log_factor=summation)
        return factor

    def sum(self, var):
        """
        Get the other variables besides the one being passed in. Group by those
        variables. Take the sum.

        Parameters:
            var: string
                The variable to be summed out.

        Returns: LogFactorAdapter
        """

        variables = var

        if isinstance(var, str):
            variables = [var]

        df = self.get_df()

        other_vars = list(set(self.get_variables()) - set(variables))

        if not other_vars:
            return FactorOne()

        return_df = df.groupby(other_vars).sum()[["value"]].reset_index()

        return LogFactorAdapter(
            data=self.data_class(
                return_df,
                storage_folder=self.log_factor.get_data().get_storage_folder(),
            )
        )

    def normalize(self, variables=None):
        """
        Make sure the values represent probabilities.

        Parameters:
            variables: list[str]
                The variables in the denominator.

        Returns: LogFactorAdapter
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
                storage_folder=self.log_factor.get_data().get_storage_folder(),
            )
        )

    def get_df(self):
        """
        Exponentiates the LogFactor values.

        Returns: pd.DataFrame
        """

        df = self.data.read()
        df["value"] = np.exp(df["value"])
        return df

    def get_data(self):
        """
        Return the data object
        """
        return self.data
