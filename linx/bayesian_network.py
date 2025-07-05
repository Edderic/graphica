"""
Bayesian Network class
"""
import pandas as pd
import numpy as np

from .conditional_probability_table import ConditionalProbabilityTable as CPT
from .directed_acyclic_graph import DirectedAcyclicGraph
from .markov_network import MarkovNetwork
from .factor import Factor
from .null_graphviz_dag import NullGraphvizDag
from .particles.particle import Particle


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

    def add_prior(self, cpt):
        """
        Add a conditional probability table. This adds a node.

        Parameters
            cpt: ConditionalProbabilityTable
        """
        self.add_node(cpt)

    def set_priors(self, dictionary, data_class, data_storage_folder=None):
        """
        Parameters:
            dictionary: dict
                Ex: {
                    'prior_var_a': {
                        'value_it_can_take_1': 0.2,
                        'value_it_can_take_2': 0.3,
                        ...
                    }
                    'prior_var_b': {
                        'value_it_can_take_1': 0.4,
                        'value_it_can_take_2': 0.2,
                        ...
                    }
                }
        """
        for prior_var, mapping in dictionary.items():
            collection = []

            for value_prior_var_can_take, proba in mapping.items():
                collection.append(
                    {
                        prior_var: value_prior_var_can_take,
                        'value': proba
                    }
                )

            df = pd.DataFrame(collection)

            givens = list(set(df.columns) - {'value', prior_var})

            cpt = CPT(
                data_class(
                    df,
                    data_storage_folder
                ),
                givens=givens,
                outcomes=[prior_var]
            )

            self.add_prior(cpt)

    def add_cpt(self, cpt):
        """
        Add a conditional probability table. This in turn adds an edge.

        Parameters
            cpt: ConditionalProbabilityTable
        """
        self.add_edge(cpt)

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

    def sample(self):
        """
        Perform forward sampling from the Bayesian Network.
        
        This method samples from the network in topological order:
        1. Start with root nodes (nodes with no parents)
        2. For each node, sample from its conditional probability distribution
           given the values of its parents (which have already been sampled)
        3. Continue until all nodes have been sampled
        
        Returns:
            Particle: A particle containing sampled values for all variables
        """
        # Get topological ordering of variables
        sorted_vars = self.topological_sort()
        
        # Initialize particle to store sampled values
        particle = Particle()
        
        # Sample each variable in topological order
        for var in sorted_vars:
            if var not in self.cpts:
                raise ValueError(f"No CPT found for variable {var}")
            
            cpt = self.cpts[var]
            sampled_value = self._sample_from_cpt(cpt, particle)
            particle.set_value(var, sampled_value)
        
        return particle
    
    def _sample_from_cpt(self, cpt, particle):
        """
        Sample a value from a conditional probability table.
        
        Parameters:
            cpt: ConditionalProbabilityTable
                The CPT to sample from
            particle: Particle
                The current particle containing already sampled values
                
        Returns:
            The sampled value for the outcome variable
        """
        # Get the data from the CPT
        df = cpt.get_data().read()
        
        # Get the outcome variable (should be only one for Bayesian Networks)
        outcomes = cpt.get_outcomes()
        if len(outcomes) != 1:
            raise ValueError(f"Expected exactly one outcome variable, got {len(outcomes)}")
        outcome_var = outcomes[0]
        
        # Get the given variables (parents)
        givens = cpt.get_givens()
        
        # Filter the dataframe based on the values of the given variables
        if givens:
            # Create a filter condition for each given variable
            for given_var in givens:
                if not particle.has_variable(given_var):
                    raise ValueError(f"Parent variable {given_var} not yet sampled")
                
                given_value = particle.get_value(given_var)
                df = df[df[given_var] == given_value]
            
            # Check if we have any rows after filtering
            if df.empty:
                raise ValueError(f"No matching rows found for given values: {[(var, particle.get_value(var)) for var in givens]}")
        
        # Extract the possible values and their probabilities
        possible_values = df[outcome_var].values
        probabilities = df['value'].values
        
        # Normalize probabilities to ensure they sum to 1
        probabilities = probabilities / probabilities.sum()
        
        # Sample from the categorical distribution
        sampled_value = np.random.choice(possible_values, p=probabilities)
        
        return sampled_value
