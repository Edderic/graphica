"""
MarkovNetwork class.
"""

from .factors import Factors


class MarkovNetwork:
    """
    Markov network. A data structure that has undirected edges. Each clique
    represents a factor.
    """

    def __init__(self):
        self.factors = {}

    def __repr__(self):
        return f"MarkovNetwork({self.factors})"

    def add_factor(self, factor):
        """
        Add factor.

        Parameters:
            factor: Factor
                Something that responds to get_variables.
        """
        variables = factor.get_variables()
        for var in variables:
            if var not in self.factors:
                self.factors[var] = Factors([])
            self.factors[var].append(factor)

    def get_factors(self, node=None):
        """
        If node is None, returns all factors.

        Parameters:
            node: string or None

        Returns: list[Factor]
        """
        cache = {}
        factors = Factors([])

        if node:
            return Factors(list(self.factors[node]))

        for _, fs in self.factors.items():
            for factor in fs:
                variables_key = "-".join(sorted(factor.get_variables()))
                if variables_key not in cache:
                    cache[variables_key] = 1
                    factors.append(factor)

        return factors

    def get_variables(self):
        """
        Get all the variables

        Returns: list[string]
        """

        return list(self.factors.keys())

    def remove_factor(self, factor):
        """
        Remove a factor.

        Returns: list[string]
        """
        variables = factor.get_variables()
        for node in variables:
            factors = self.factors[node]
            factors.remove(factor)

    def apply_query(self, query):
        """
        Replaces each factor with a filtered version of the factor, if the
        query applies to it.

        Parameters:
            query: Query
        """
        filters = query.get_filters()

        for obj in filters:
            variable = list(obj.keys())[0]
            factors = self.get_factors(variable)

            for factor in factors:
                self.add_factor(factor.filter(query))
                self.remove_factor(factor)

    def to_markov_network(self):
        """
        Returns a copy of this markov network.

        Returns: MarkovNetwork
        """
        # TODO: solve this
