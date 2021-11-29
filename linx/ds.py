"""
Data Structures module.

Classes:
    - ConditionalProbabilityTable
    - DirectedAcyclicGraph
    - Factor
    - Factors
"""


class ConditionalProbabilityTable:
    """
    Conditional Probability Table class. Meant to be used to represent
    conditional probabilities for Bayesian Networks.
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, df, given, outcomes):
        self.df = df
        self.given = given
        self.outcomes = outcomes

        self.__validate__()

    def __validate__(self):
        existing_cols = self.df.reset_index().columns

        if 'count' not in existing_cols:
            raise ValueError("The column 'count' must exist.")

        given_plus_outcomes_cols = set(self.given + self.outcomes)

        if given_plus_outcomes_cols.intersection(
            set(existing_cols) - {'count'}
        ) != given_plus_outcomes_cols:

            raise ValueError(
                "Mismatch between dataframe columns {existing_cols} and"
                + " given and outcomes {given_plus_outcomes_cols}"
            )


class DirectedAcyclicGraph:
    """
    Directed Acyclic Graph. Useful for representing Bayesian Networks.
    """
    def __init__(self):
        self.data_structure = {}

    def add_edge(self, start, end):
        """
        Add edge from start node to end node.

        Parameters:
            start: str
                Start node.
            end: str
                End node.
        """

        if start not in self.data_structure:
            self.data_structure[start] = [end]
        elif end not in self.data_structure[start]:
            self.data_structure[start].append(end)

    def get_neighbors(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """

        return self.get_parents(node) + self.get_children(node)

    def get_parents(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """
        parents = []

        for other_node, children in self.data_structure.items():
            if node in children:
                parents.append(other_node)

        return parents

    def get_children(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """

        if node not in self.data_structure:
            return []

        return self.data_structure[node]


class Factor:
    """
    Class for representing factors.
    """
    def __init__(self, df=None, cpt=None):
        if df is not None:
            self.df = df
        else:
            self.df = cpt.df

    def get_variables(self):
        """
        Return variables
        """
        return list(set(self.df.columns) - {'count'})

    def div(self, other):
        """
        Parameters:
            other: Factor

        Returns: Factor
        """

        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        merged = self.df.merge(other.df, on=common)
        merged['count'] = merged.count_x / merged.count_y

        return Factor(
            merged[
                list(left_vars.union(right_vars.union({'count'})))
            ]
        )


    def prod(self, other):
        """
        Parameters:
            other: Factor

        Returns: Factor
        """

        left_vars = set(list(self.get_variables()))
        right_vars = set(list(other.get_variables()))
        common = list(
            left_vars.intersection(right_vars)
        )

        merged = self.df.merge(other.df, on=common)
        merged['count'] = merged.count_x * merged.count_y

        return Factor(
            merged[
                list(left_vars.union(right_vars.union({'count'})))
            ]
        )

    def sum(self, var):
        """
        Parameters:
            var: string
                The variable to be summed out.

        Returns: Factor
        """

        other_vars = list(set(self.get_variables()) - {'count', var})
        new_df = self.df.groupby(other_vars)\
            .sum()[['count']].reset_index()

        return Factor(new_df)
