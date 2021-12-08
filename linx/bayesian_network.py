"""
Bayesian Network class
"""
from .directed_acyclic_graph import DirectedAcyclicGraph
from .markov_network import MarkovNetwork
from .factor import Factor
from .null_graphviz_dag import NullGraphvizDag


class BayesianNetwork(DirectedAcyclicGraph):
    """
    Bayesian Network that stores ConditionalProbabilityTables.

    Parameters:
        cpts: list[ConditionalProbabilityTable]. Optional.
            Meant for specifying conditional probability tables of variables
            that are endogenous..

        priors: list[ConditionalProbabilityTable]. Optional.
            Meant for probability tables of Variables that are exogenous.

        graphviz_dag: DiGraph
            Could be used to display the graph.
    """

    def __init__(self, cpts=None, priors=None, graphviz_dag=None):
        super().__init__()
        if graphviz_dag is None:
            self.graphviz_dag = NullGraphvizDag()
        else:
            self.graphviz_dag = graphviz_dag

        if cpts is None:
            self.cpts = {}
        else:
            self.cpts = {}
            for cpt in cpts:
                self.add_edge(cpt)

        if priors:
            for prior_cpt in priors:
                self.add_node(prior_cpt)

    def __repr__(self):
        return f"BayesianNetwork(\n\t{self.cpts})"

    def add_node(self, cpt):
        """
        Add a conditional probability table. This adds a node.

        Parameters:
            cpt: ConditionalProbabilityTable
        """
        outcomes = cpt.get_outcomes()
        if cpt.get_givens():
            raise ValueError(
                "There should not be any givens for the CPT when adding a"
                + " node."
            )

        if len(outcomes) != 1:
            raise ValueError(
                "There should only be one outcome for a CPT of a "
                + "Bayesian Network."
            )

        for outcome in outcomes:
            self.cpts[outcome] = cpt

            self.graphviz_dag.node(outcome)
            super().add_node(outcome)

    def add_edge(self, cpt):
        """
        Add a conditional probability table. This in turn adds an edge.

        Parameters:
            cpt: ConditionalProbabilityTable
        """
        outcomes = cpt.get_outcomes()
        givens = cpt.get_givens()

        if len(outcomes) != 1:
            raise ValueError(
                "There should only be one outcome for a CPT of a "
                + "Bayesian Network."
            )

        for outcome in outcomes:
            self.cpts[outcome] = cpt

            for given in givens:
                self.graphviz_dag.edge(given, outcome)
                super().add_edge(start=given, end=outcome)

    def find_cpt_for_node(self, node):
        """
        Find conditional probability table for node.

        Parameters:
            node: str

        Returns: ConditionalProbabilityTable
        """

        return self.cpts[node]

    def to_markov_network(self):
        """
        Returns: MarkovNetwork
        """
        markov_network = MarkovNetwork()
        for _, cpt in self.cpts.items():
            factor = Factor(cpt=cpt)
            markov_network.add_factor(factor)

        return markov_network
