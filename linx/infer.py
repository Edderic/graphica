"""
Algorithms for inference.

Classes:
    VariableElimination

Functions:
    min_neighbors
"""
from .debug import log_debug


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
        query,
        greedy_heuristic=None,
    ):
        # TODO: handle queries of do(x)
        # TODO: Maybe have "outcomes" and "given" be wrapped into a "Query"
        # object.
        self.network = network.to_markov_network()
        self.query = query
        if greedy_heuristic is None:
            self.greedy_heuristic = min_fill_edges
        else:
            self.greedy_heuristic = greedy_heuristic

    def __repr__(self):
        return f"VariableElimination({self.network})"

    def compute(self):
        """
        Runs the variable elimination algorithm.

        Returns: Factor
            A factor that represents the query.
        """
        self.network.apply_query(self.query)

        numerator_eliminateables = list(
            set(self.network.get_variables())
            - set(self.query.get_outcome_variables())
            .union(set(self.query.get_given_variables()))
        )

        self.__compute__(
            numerator_eliminateables
        )

        numer_factors = self.network.get_factors()
        numerator_prod = numer_factors.prod()

        left_to_eliminate = list(
            set(self.query.get_outcome_variables()) -
            set(self.query.get_given_variables())
        )

        self.__compute__(
            left_to_eliminate
        )

        denom_prod = self.network.get_factors().prod()

        return numerator_prod.div(denom_prod)

    def __compute__(self, eliminateables):
        while eliminateables:
            log_debug(
                {
                    'msg': 'top of eliminateables',
                    'eliminateables': eliminateables
                }
            )
            best_eliminateable, _ = self.greedy_heuristic(
                eliminateables=eliminateables,
                network=self.network
            )

            log_debug(
                {
                    'msg': 'best eliminateable',
                    'best_eliminateable': best_eliminateable,
                    'min': _
                }
            )

            factors = self.network.get_factors(best_eliminateable)
            log_debug(
                {
                    'msg': 'before prod',
                    'factors for best eliminateable': factors
                }
            )
            factor_prod = factors.prod()

            log_debug(
                {
                    'msg': 'after prod',
                    'factor_prod': factor_prod
                }
            )
            # Update network with new factor
            new_factor = factor_prod.sum(best_eliminateable)

            log_debug(
                {
                    'msg': 'after summing',
                    'new_factor': new_factor
                }
            )
            self.network.add_factor(
                factor=new_factor
            )

            log_debug(
                {
                    'msg': 'after adding',
                    'network': self.network
                }
            )

            # Remove old factors
            for factor in factors:
                self.network.remove_factor(
                    factor=factor
                )

            log_debug(
                {
                    'msg': 'after remove_factor',
                    'network': self.network
                }
            )

            eliminateables = list(
                set(eliminateables) - {best_eliminateable}
            )
