"""
DefaultTransition class for Metropolis-Hastings sampling.
"""

from ..random.deterministic import Deterministic


# pylint: disable=too-few-public-methods
class DefaultTransition:
    """
    Default transition function for Metropolis-Hastings sampling.

    This class provides intelligent transition logic that:
    1. Respects Query constraints (fixed values and filters)
    2. Updates Deterministic node chains when their parents change
    3. Uses appropriate perturbation strategies for different distribution types

    Parameters:
        bayesian_network: BayesianNetwork
            The Bayesian network to transition.
        query: Query
            Query object specifying outcomes and givens.
    """

    def __init__(self, bayesian_network, query):
        self.bayesian_network = bayesian_network
        self.query = query

        # Validate query variables exist in network
        self._validate_query()

        # Cache dependency graph for Deterministic nodes
        self._cache_deterministic_dependencies()

        # Cache topological sort of variables
        self.sorted_vars = self.bayesian_network.topological_sort()

    def _validate_query(self):
        """Validate that all variables in the query exist in the network."""
        network_vars = set(self.bayesian_network.random_variables.keys())

        # Check outcomes
        for outcome in self.query.get_outcome_variables():
            if outcome not in network_vars:
                raise ValueError(
                    f"Outcome variable '{outcome}' not found in Bayesian network"
                )

        # Check givens
        for given in self.query.get_given_variables():
            if given not in network_vars:
                raise ValueError(
                    f"Given variable '{given}' not found in Bayesian network"
                )

    def _cache_deterministic_dependencies(self):
        """Cache which Deterministic nodes depend on which other nodes."""
        self.deterministic_nodes = {}
        self.deterministic_children = {}

        # Find all Deterministic nodes
        for var_name, rv in self.bayesian_network.random_variables.items():
            if not isinstance(rv, Deterministic):
                continue

            self.deterministic_nodes[var_name] = rv

            # Find children of this Deterministic node
            children = []
            for (
                child_name,
                child_rv,
            ) in self.bayesian_network.random_variables.items():
                if hasattr(child_rv, "get_parents"):
                    parents = child_rv.get_parents()
                    for _, parent in parents.items():
                        if hasattr(parent, "name") and parent.name == var_name:
                            children.append(child_name)
                            break

            self.deterministic_children[var_name] = children

    def _get_deterministic_chain(self, start_var):
        """
        Get all Deterministic nodes that depend on start_var (including
        start_var if it's Deterministic).
        """
        chain = set()
        to_process = [start_var]

        while to_process:
            current_var = to_process.pop(0)
            if current_var in chain:
                continue

            chain.add(current_var)

            # Add Deterministic children to processing queue
            if current_var in self.deterministic_children:
                for child in self.deterministic_children[current_var]:
                    if child in self.deterministic_nodes:
                        to_process.append(child)

        return list(chain)

    def _perturb_value(self, var_name, current_value, particle):
        """Perturb a value using the distribution's perturb method with parent values."""
        rv = self.bayesian_network.random_variables[var_name]

        # Gather parent values for the random variable
        parent_values = {}
        for parent_name, parent in rv.get_parents().items():
            if hasattr(parent, "name"):
                parent_values[parent_name] = particle.get_value(parent.name)
            else:
                # Handle case where parent might be a fixed parameter
                parent_values[parent_name] = parent

        # Call the distribution's perturb method with parent values
        return rv.perturb(current_value, **parent_values)

    def _check_filters(self, particle):
        """Check if particle satisfies all filters in the query."""
        for given in self.query.givens:
            if isinstance(given, dict):
                var_name = list(given.keys())[0]
                filter_value = given[var_name]

                if callable(filter_value):
                    # Apply filter function
                    if not filter_value(particle):
                        return False
                else:
                    # Fixed value - should already be set correctly
                    if particle.get_value(var_name) != filter_value:
                        return False

        return True

    def _transition(self, particle, new_particle):
        # Get given values that should be fixed
        given_values = self.query.get_given_values()

        # Transition all variables except those fixed in givens
        for var_name in self.sorted_vars:
            if var_name in given_values:
                # Skip variables that are fixed in givens
                continue

            if not particle.has_variable(var_name):
                # Skip variables not in current particle
                continue

            current_value = particle.get_value(var_name)

            # Perturb the value
            new_value = self._perturb_value(var_name, current_value, new_particle)
            new_particle.set_value(var_name, new_value)

    def _setup_new_particle(self, particle):
        # Create new particle
        new_particle = particle.copy()

        # Get given values that should be fixed
        given_values = self.query.get_given_values()

        # Set fixed values from givens (only if not callable)
        for var_name, value in given_values.items():
            if not callable(value):
                new_particle.set_value(var_name, value)

        return new_particle

    def transition(self, particle):
        """
        Perform a transition step.

        Parameters:
            particle: Particle
                Current particle to transition from.

        Returns:
            Particle: New particle after transition.
        """
        new_particle = self._setup_new_particle(particle)

        self._transition(particle, new_particle)

        # Check if new particle satisfies filters
        if not self._check_filters(new_particle):
            # If filters not satisfied, return original particle
            return particle

        return new_particle
