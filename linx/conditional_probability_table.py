"""
ConditionalProbabilityTable class
"""
from .errors import ArgumentError


class ConditionalProbabilityTable:
    """
    Conditional Probability Table class. Meant to be used to represent
    conditional probabilities for Bayesian Networks.

    Parameters:
        data: Data
            A data object.

        outcomes: list[str]
            In P(X,Y | Z,A), this is the left side (i.e. X, Y).

        givens: list[str]
            In P(X,Y | Z,A), this is the right side (i.e. Z, A).
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, data, outcomes, givens=None):
        if givens is None:
            givens = []

        self.givens = givens
        self.outcomes = outcomes
        self.data = data

        self.__validate__()

    def __repr__(self):
        return f"ConditionalProbabilityTable(\n\tgivens: {self.givens},"\
            + f"\n\toutcomes: {self.outcomes}\n\tdf:\n\t\n{self.data.read()})"

    def __validate__(self):
        existing_cols = self.data.read().reset_index().columns

        if 'value' not in existing_cols:
            raise ValueError("The column 'value' must exist.")

        given_plus_outcomes_cols = set(self.givens + self.outcomes)

        if given_plus_outcomes_cols.intersection(
            set(existing_cols) - {'value'}
        ) != given_plus_outcomes_cols:

            raise ArgumentError(
                f"Mismatch between dataframe columns {existing_cols} and"
                + f" given and outcomes {given_plus_outcomes_cols}"
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
