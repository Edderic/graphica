"""
Bayesian Network class
"""
import numpy as np
from .directed_acyclic_graph import DirectedAcyclicGraph
from .particles.particle import Particle

class BayesianNetwork(DirectedAcyclicGraph):
    """
    Bayesian Network that stores RandomVariables (including CPTs).

    Parameters:
        random_variables: dict[str, RandomVariable]. Optional.
            Dictionary mapping variable names to RandomVariable objects.
    """
    def __init__(self, random_variables=None):
        super().__init__()
        if random_variables is None:
            self.random_variables = {}
        else:
            self.random_variables = random_variables.copy()

    def add_node(self, rv):
        """
        Add a random variable to the network.
        Parameters:
            rv: RandomVariable
                The random variable to add. Must have a name.
        """
        if rv.name is None:
            raise ValueError("Random variable must have a name")
        self.random_variables[rv.name] = rv
        super().add_node(rv.name)
        # Add edges for parent relationships
        for parent in rv.get_parents():
            if parent.name:
                self.add_edge(parent.name, rv.name)

    def add_edge(self, parent_name, child_name):
        """
        Add an edge from parent to child in the DAG.
        Parameters:
            parent_name: str
            child_name: str
        """
        super().add_edge(parent_name, child_name)

    def get_random_variables(self):
        """
        Get all random variables in the network.
        Returns:
            dict[str, RandomVariable]: Dictionary of random variables.
        """
        return self.random_variables.copy()

    def __repr__(self):
        return f"BayesianNetwork(\n\t{self.random_variables}\n)"

    def sample(self):
        """
        Perform forward sampling from the Bayesian Network.
        Samples each variable in topological order, using parent values as needed.
        Returns:
            Particle: A particle containing sampled values for all variables
        """
        sorted_vars = self.topological_sort()
        particle = Particle()
        for var in sorted_vars:
            rv = self.random_variables[var]
            # Gather parent values
            parent_values = {}
            for parent in rv.get_parents():
                if not particle.has_variable(parent.name):
                    raise ValueError(f"Parent variable {parent.name} not yet sampled")
                parent_values[parent.name] = particle.get_value(parent.name)
            # Sample from the random variable
            sampled_value = rv.sample(**parent_values)
            # If sample returns a dict (CPT), extract the value for this variable
            if isinstance(sampled_value, dict) and var in sampled_value:
                particle.set_value(var, sampled_value[var])
            else:
                particle.set_value(var, sampled_value)
        return particle
