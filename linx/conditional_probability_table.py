"""
ConditionalProbabilityTable class
"""
from .errors import ArgumentError


class ConditionalProbabilityTable:
    """
    Conditional Probability Table class. Meant to be used to represent
    conditional probabilities for Bayesian Networks.
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, df, outcomes, givens=None):
        self.df = df

        if givens is None:
            givens = []

        self.givens = givens
        self.outcomes = outcomes

        self.__validate__()

    def __repr__(self):
        return f"ConditionalProbabilityTable(\n\tgivens: {self.givens},"\
                +f"\n\toutcomes: {self.outcomes}\n\tdf:\n\t\n{self.df})"
    def __validate__(self):
        existing_cols = self.df.reset_index().columns

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
