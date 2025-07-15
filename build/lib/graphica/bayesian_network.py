"""
Bayesian Network class
"""
import numpy as np
from .directed_acyclic_graph import DirectedAcyclicGraph
from .particles.particle import Particle
from .conditional_probability_table import ConditionalProbabilityTable

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

    def add_nodes(self, rvs):
        if isinstance(rvs, dict):
            # If rvs is a dict, loop through the values
            for rv in rvs.values():
                self.add_node(rv)
        else:
            # If rvs is a list or other iterable, loop through directly
            for rv in rvs:
                self.add_node(rv)

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

        # For CPTs, add edges based on givens
        if hasattr(rv, 'get_givens'):
            for parent_name in rv.get_givens():
                self.add_edge(parent_name, rv.name)
        else:
            # Add edges for parent relationships
            for parent_name, parent in rv.get_parents().items():
                if parent is not None and hasattr(parent, 'name'):
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

            # For CPTs, get parent values from givens
            if hasattr(rv, 'get_givens'):
                for parent_name in rv.get_givens():
                    if not particle.has_variable(parent_name):
                        raise ValueError(f"Parent variable {parent_name} not yet sampled")
                    parent_val = particle.get_value(parent_name)
                    # TODO: this is a code smell
                    # Discretize parent values for CPTs
                    if parent_name in self.random_variables:
                        parent_rv = self.random_variables[parent_name]
                        if parent_rv.__class__.__name__ == 'Uniform':
                            parent_values[parent_name] = int(parent_val > 0.5)
                        elif parent_rv.__class__.__name__ == 'Normal':
                            parent_values[parent_name] = int(parent_val > 0)
                        else:
                            parent_values[parent_name] = parent_val
                    else:
                        parent_values[parent_name] = parent_val
            else:
                # For other random variables, get parent values from parent objects
                for parent_name_from_child, parent in rv.get_parents().items():
                    parent_values[parent_name_from_child] = particle.get_value(parent.name)

            # Sample from the random variable
            sampled_value = rv.sample(**parent_values)

            # If sample returns a dict (CPT), extract the value for this variable
            if isinstance(sampled_value, dict) and var in sampled_value:
                particle.set_value(var, sampled_value[var])
            else:
                particle.set_value(var, sampled_value)
        return particle

    def to_markov_network(self):
        """
        Convert Bayesian Network to Markov Network.

        This method assumes all random variables are ConditionalProbabilityTables.
        Each CPT is converted to a Factor and added to the Markov Network.

        Returns:
            MarkovNetwork: The converted Markov Network

        Raises:
            ValueError: If any random variable is not a ConditionalProbabilityTable
        """
        from .markov_network import MarkovNetwork
        from .factor import Factor

        # Check that all random variables are CPTs
        for var_name, rv in self.random_variables.items():
            if not isinstance(rv, ConditionalProbabilityTable):
                raise ValueError(
                    f"Random variable '{var_name}' is not a ConditionalProbabilityTable. "
                    f"Found: {type(rv).__name__}. "
                    "to_markov_network() only works with CPTs."
                )

        # Create Markov Network
        markov_network = MarkovNetwork()

        # Convert each CPT to a Factor and add to Markov Network
        for rv in self.random_variables.values():
            factor = Factor(cpt=rv)
            markov_network.add_factor(factor)

        return markov_network
