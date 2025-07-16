"""
ConditionalProbabilityTable class
"""

import numpy as np
import pandas as pd
from .errors import ArgumentError
from .random.random_variable import RandomVariable


class ConditionalProbabilityTable(RandomVariable):
    """
    Conditional Probability Table class. Meant to be used to represent
    conditional probabilities for Bayesian Networks.

    Parameters:
        data: Data
            A data object.

        outcomes: list[str]
            In P(X,Y | Z,A), this is the left side (i.e. X, Y).

        givens: list[str]
            In P(X,Y | Z,A), this is the right side (i.e. Z, A). If None, this is assumd to be an empty list.
    """

    # pylint:disable=too-few-public-methods
    def __init__(
        self, data=None, table=None, outcomes=None, givens=None, name=None, **kwargs
    ):
        # Call parent constructor first
        super().__init__(name=name, **kwargs)

        if name is None and len(outcomes) == 1:
            self.name = outcomes[0]
        # Process CPT-specific parameters
        if givens is None:
            givens = []

        if outcomes is None:
            raise ValueError("outcomes parameter is required")

        # Handle table parameter
        if table is not None:
            if data is not None:
                raise ValueError("Cannot specify both 'data' and 'table' parameters")

            # Convert table to DataFrame and create InMemoryData
            df = pd.DataFrame(table)
            from .data import InMemoryData

            data = InMemoryData(df)

        if data is None:
            raise ValueError("Either 'data' or 'table' parameter must be provided")

        self.givens = givens
        self.outcomes = outcomes
        self.data = data

        self.__validate__()

    def __repr__(self):
        return (
            f"ConditionalProbabilityTable(\n\tgivens: {self.givens},"
            + f"\n\toutcomes: {self.outcomes}\n\tdf:\n\t\n{self.data.read()})"
        )

    def __validate__(self):
        existing_cols = list(set(self.data.read().reset_index().columns) - {"index"})

        if "value" not in existing_cols:
            raise ValueError("The column 'value' must exist.")

        given_plus_outcomes_cols = set(self.givens + self.outcomes)

        intersection = given_plus_outcomes_cols.intersection(
            set(existing_cols) - {"value", "index"}
        )

        if intersection != given_plus_outcomes_cols:
            raise ArgumentError(
                "Mismatch between dataframe columns: "
                + f"\n\n\t{existing_cols}\n\n and"
                + f" given and outcomes \n\n\t{given_plus_outcomes_cols}\n\n"
                + "given_plus_outcomes_cols - intersection: \n\n"
                + f"\t{set(given_plus_outcomes_cols) - set(intersection)}"
            )

    def get_data(self):
        """
        Returns: Data
        """

        return self.data

    def get_givens(self):
        """
        Returns list[str]
            List of variable names that are being conditioned on.
        """
        return self.givens

    def get_outcomes(self):
        """
        Returns list[str]
            List of variable names in the left side of the query.
        """
        return self.outcomes

    def sample_with_given_values(self, given_values=None):
        """
        Sample a value from this conditional probability table.

        Parameters:
            given_values: dict, optional
                Dictionary mapping given variable names to their values.
                If None, assumes no given variables (prior distribution).

        Returns:
            dict: Dictionary mapping outcome variable names to their sampled values
        """
        # Get the data from the CPT
        df = self.get_data().read()

        # Get the outcome variables
        outcomes = self.get_outcomes()

        # Get the given variables (parents)
        givens = self.get_givens()

        # Filter the dataframe based on the values of the given variables
        if givens:
            if not given_values:
                raise ValueError(
                    f"Given variables {givens} are required but no given_values provided"
                )

            # Create a filter condition for each given variable
            for given_var in givens:
                if given_var not in given_values:
                    raise ValueError(
                        f"Given variable {given_var} not provided in given_values"
                    )

                given_value = given_values[given_var]
                df = df[df[given_var] == given_value]

            # Check if we have any rows after filtering
            if df.empty:
                raise ValueError(
                    f"No matching rows found for given values: {given_values}"
                )

        # Sample from each outcome variable
        sampled_values = {}
        for outcome_var in outcomes:
            # Extract the possible values and their probabilities for this outcome
            possible_values = df[outcome_var].values
            probabilities = df["value"].values

            # Normalize probabilities to ensure they sum to 1
            probabilities = probabilities / probabilities.sum()

            # Sample from the categorical distribution
            sampled_value = np.random.choice(possible_values, p=probabilities)
            sampled_values[outcome_var] = sampled_value

        return sampled_values

    def pdf(self, x, **kwargs):
        """
        Probability mass function for discrete variables.

        Parameters:
            x: dict or array-like
                Values for the outcome variables.
            **kwargs: dict
                Values for the given variables (parents).

        Returns:
            float: Probability mass at the given point.
        """
        # Handle different input formats
        if isinstance(x, dict):
            outcome_values = x
        else:
            # Assume x is array-like and corresponds to outcomes in order
            outcome_values = dict(zip(self.outcomes, x))

        # Get given values from kwargs
        given_values = {var: kwargs[var] for var in self.givens if var in kwargs}

        # Get the data from the CPT
        df = self.get_data().read()

        # Filter the dataframe based on the values
        for var, value in list(outcome_values.items()) + list(given_values.items()):
            df = df[df[var] == value]

        if df.empty:
            return 0.0

        # Return the probability value
        return df["value"].iloc[0]

    def logpdf(self, x, **kwargs):
        """
        Logarithm of the probability mass function.

        Parameters:
            x: dict or array-like
                Values for the outcome variables.
            **kwargs: dict
                Values for the given variables (parents).

        Returns:
            float: Log probability mass at the given point.
        """
        prob = self.pdf(x, **kwargs)
        if prob <= 0:
            return -np.inf
        return np.log(prob)

    def sample(self, size=None, **kwargs):
        """
        Sample from the conditional probability table.

        Parameters:
            size: int or tuple of ints, optional
                Output shape. If None, returns a single sample.
            **kwargs: dict
                Values for the given variables (parents).

        Returns:
            dict or list[dict]: Sampled values for outcome variables.
        """
        if size is None:
            return self._sample_single(**kwargs)
        else:
            # Handle multiple samples
            if isinstance(size, int):
                return [self._sample_single(**kwargs) for _ in range(size)]
            else:
                # Handle tuple sizes
                total_samples = np.prod(size)
                samples = [self._sample_single(**kwargs) for _ in range(total_samples)]
                return np.array(samples).reshape(size)

    def _sample_single(self, **kwargs):
        """
        Sample a single value from the conditional probability table.

        Parameters:
            **kwargs: dict
                Values for the given variables (parents).

        Returns:
            dict: Sampled values for outcome variables.
        """
        return self.sample_with_given_values(given_values=kwargs)
