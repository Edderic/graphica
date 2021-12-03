"""
Query class
"""


class Query:
    """
    Query object. Outcomes refers to the left-hand side of a probability
    expression, while the given refers to the right-hand side.

    i.e. P(Outcomes | Givens)
    distribution.

    Parameters:
        outcomes: list[string or dict]
            If string, this is a variable name.
            If dict, then this is a key-value pair.
                Key: string
                    The variable name.
                Value: callable
                    Used for filtering.
                    Ex: {
                        'some_var': lambda x: (
                            x['some_var'] > 20
                        ) & (x['some_var'] < 30)
                    }
        givens: list[string or dict]
            If string, this is a variable name.
            If dict, then this is a key-value pair.
                Key: string
                    The variable name.
                Value: callable
                    Used for filtering.
                    Ex: {
                        'some_var': lambda x: (
                            x['some_var'] > 20
                        ) & (x['some_var'] < 30)
                    }
    """
    def __init__(
        self,
        outcomes,
        givens=None
    ):

        if givens is None:
            givens = []
        else:
            self.givens = givens

        self.outcomes = outcomes
        self.outcome_vars = []
        self.given_vars = []
        self.filters = []

    def get_filters(self):
        """
        Get filters

        Returns: list[dict]
        """
        if self.filters:
            return self.filters

        for outcome in self.outcomes:
            if isinstance(outcome, dict):
                self.filters.append(outcome)

        for given in self.givens:
            if isinstance(given, dict):
                self.filters.append(given)

        return self.filters

    def get_outcome_variables(self):
        """
        Get outcome variables

        Returns: list[str]
        """
        if self.outcome_vars:
            return self.outcome_vars

        for outcome in self.outcomes:
            if isinstance(outcome, str):
                self.outcome_vars.append(outcome)
            elif isinstance(outcome, dict):
                self.outcome_vars.append(list(outcome.keys())[0])
            else:
                raise TypeError(
                    f'Unrecognized type of object {outcome} for query'
                )

        return self.outcome_vars

    def get_given_variables(self):
        """
        Get given variables.

        Returns: list[str]
        """
        if self.given_vars:
            return self.given_vars

        for given in self.givens:
            if isinstance(given, str):
                self.given_vars.append(given)
            elif isinstance(given, dict):
                self.given_vars.append(list(given.keys())[0])
            else:
                raise TypeError(
                    f'Unrecognized type of object {given} for query'
                )

        return self.given_vars
