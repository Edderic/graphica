"""
ConditionalProbabilityTable class
"""
import numpy as np
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
        existing_cols = list(
            set(self.data.read().reset_index().columns)
            - {'index'}
        )

        if 'value' not in existing_cols:
            raise ValueError("The column 'value' must exist.")

        given_plus_outcomes_cols = set(self.givens + self.outcomes)

        intersection = given_plus_outcomes_cols.intersection(
            set(existing_cols) - {'value', 'index'}
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

    def sample(self, given_values=None):
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
                raise ValueError(f"Given variables {givens} are required but no given_values provided")
            
            # Create a filter condition for each given variable
            for given_var in givens:
                if given_var not in given_values:
                    raise ValueError(f"Given variable {given_var} not provided in given_values")
                
                given_value = given_values[given_var]
                df = df[df[given_var] == given_value]
            
            # Check if we have any rows after filtering
            if df.empty:
                raise ValueError(f"No matching rows found for given values: {given_values}")
        
        # Sample from each outcome variable
        sampled_values = {}
        for outcome_var in outcomes:
            # Extract the possible values and their probabilities for this outcome
            possible_values = df[outcome_var].values
            probabilities = df['value'].values
            
            # Normalize probabilities to ensure they sum to 1
            probabilities = probabilities / probabilities.sum()
            
            # Sample from the categorical distribution
            sampled_value = np.random.choice(possible_values, p=probabilities)
            sampled_values[outcome_var] = sampled_value
        
        return sampled_values
