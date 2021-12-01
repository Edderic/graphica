"""
Algorithms for inference.

Classes:
    VariableElimination

Functions:
    min_neighbors
"""


def min_fill_edges(eliminateables, network):
    """
    A greedy heuristic meant to prevent exponential blow up of factors for
    VariableElimination.

    Parameters:
        eliminateables: list[str]
            Variables to eliminate.

        network: MarkovNetwork

    Returns: tuple (string, integer)
        First item is the best choice to eliminate.

        Second item is the min number of variables associated with the best
        choice.
    """
    min_number_of_variables = len(network.get_variables())
    best_choice = None

    for eliminateable in eliminateables:
        factors = network.get_factors(eliminateable)
        num_vars = len(factors.get_variables())

        if min_number_of_variables > num_vars:
            min_number_of_variables = num_vars
            best_choice = eliminateable

    return best_choice, min_number_of_variables


class VariableElimination:
    """
    Algorithm that makes use of dynamic programming.

    Parameters:
        network: MarkovNetwork

        outcomes: list[str]
            Ex: P(X,Y,Z | A, B, C)

            The left side of the conditioning bar is the "outcomes" section.

        given: list[str]
            Ex: P(X,Y,Z | A, B, C)

            The right side of the conditioning bar is the "given" section.

        greedy_heuristic: callable. Defaults to min_fill_edges
            A callable that has two arguments:
                eliminateables: The list of variables to eliminate.
                network: MarkovNetwork

    """
    def __init__(
        self,
        network,
        outcomes,
        given,
        greedy_heuristic=None
    ):
        # TODO: handle queries of do(x)
        # TODO: Maybe have "outcomes" and "given" be wrapped into a "Query"
        # object.
        self.network = network.to_markov_network()
        self.outcomes = outcomes
        self.given = given

        if greedy_heuristic is None:
            self.greedy_heuristic = min_fill_edges
        else:
            self.greedy_heuristic = greedy_heuristic

    def compute(self):
        """
        Runs the variable elimination algorithm.

        Returns: Factor
            A factor that represents the query.
        """
        numerator_eliminateables = list(
            set(self.network.get_variables())
            - set(self.outcomes).union(set(self.given))
        )

        self.__compute__(
            numerator_eliminateables
        )

        numer_factors = self.network.get_factors()
        numerator_prod = numer_factors.prod()

        left_to_eliminate = list(
            set(self.outcomes) - set(self.given)
        )

        self.__compute__(
            left_to_eliminate
        )

        denom_prod = self.network.get_factors().prod()

        return numerator_prod.div(denom_prod)

    def __compute__(self, eliminateables):
        while eliminateables:
            best_eliminateable, _ = self.greedy_heuristic(
                eliminateables=eliminateables,
                network=self.network
            )

            factors = self.network.get_factors(best_eliminateable)
            factor_prod = factors.prod()

            # Update network with new factor
            new_factor = factor_prod.sum(best_eliminateable)
            self.network.add_factor(
                factor=new_factor
            )

            # Remove old factors
            for factor in factors:
                self.network.remove_factor(
                    factor=factor
                )

            eliminateables = list(
                set(eliminateables) - {best_eliminateable}
            )
