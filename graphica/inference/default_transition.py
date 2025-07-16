"""
DefaultTransition class for Metropolis-Hastings sampling.
"""

import inspect

from ..random.deterministic import Deterministic
from ..random.normal import Normal
from ..random.gamma import Gamma
from ..random.beta import Beta
from ..random.binomial import Binomial
from ..random.uniform import Uniform


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
            if isinstance(rv, Deterministic):
                self.deterministic_nodes[var_name] = rv

                # Find children of this Deterministic node
                children = []
                for (
                    child_name,
                    child_rv,
                ) in self.bayesian_network.random_variables.items():
                    if hasattr(child_rv, "get_parents"):
                        parents = child_rv.get_parents()
                        for parent_name, parent in parents.items():
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

    def _update_deterministic_chain(self, particle, chain_vars):
        """Update all Deterministic nodes in the chain."""
        for var_name in chain_vars:
            if var_name in self.deterministic_nodes:
                rv = self.deterministic_nodes[var_name]

                # Gather required parameters for the Deterministic node
                sig = inspect.signature(rv.callable_func)
                required_params = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                ]

                # Get parameter values from particle
                params = {}
                for param_name in required_params:
                    if param_name in rv.fixed_params:
                        # Use fixed parameter
                        params[param_name] = rv.fixed_params[param_name]
                        continue

                    # Get from particle
                    if not particle.has_variable(param_name):
                        msg = f"Missing param '{param_name}' for Deterministic node '{var_name}'"
                        raise ValueError(
                            msg
                        )
                        params[param_name] = particle.get_value(param_name)

                # Sample new value
                try:
                    new_value = rv.sample(**params)
                    particle.set_value(var_name, new_value)
                except Exception as e:
                    raise ValueError(
                        f"Error sampling Deterministic node '{var_name}': {e}"
                    )

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

    def _transition_non_deterministic(self, particle, new_particle):
        # Get given values that should be fixed
        given_values = self.query.get_given_values()
        
        # Transition all variables except those fixed in givens
        for var_name, rv in self.bayesian_network.random_variables.items():
            if var_name in given_values:
                # Skip variables that are fixed in givens
                continue

            if not particle.has_variable(var_name):
                # Skip variables not in current particle
                continue

            # Skip Deterministic nodes - they will be updated through their parents
            if isinstance(rv, Deterministic):
                continue

            current_value = particle.get_value(var_name)

            # Perturb the value
            new_value = self._perturb_value(var_name, current_value, particle)
            new_particle.set_value(var_name, new_value)

    def _transition_deterministic_nodes(self, new_particle):
        """
        After all transitions, update all Deterministic nodes in topological order
        (to ensure all values are consistent with the current particle state)
        Use the BayesianNetwork's topological sort

        TODO: probably a good idea for this class to be a super class of
        specialized transition functions
        """
        if hasattr(self.bayesian_network, "topological_sort"):
            topo_order = self.bayesian_network.topological_sort()
        else:
            topo_order = self.bayesian_network.get_nodes()  # fallback
        for var_name in topo_order:
            if var_name in self.deterministic_nodes:
                rv = self.deterministic_nodes[var_name]

                sig = inspect.signature(rv.callable_func)
                required_params = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                ]
                params = {}
                for param_name in required_params:
                    if param_name in rv.fixed_params:
                        params[param_name] = rv.fixed_params[param_name]
                        continue

                    if not new_particle.has_variable(param_name):
                        msg = f"Missing param '{param_name}' for Deterministic node '{var_name}'"
                        raise ValueError(
                            msg
                        )
                    params[param_name] = new_particle.get_value(param_name)
                try:
                    new_value = rv.sample(**params)
                    new_particle.set_value(var_name, new_value)
                except Exception as e:
                    raise ValueError(
                        f"Error sampling Deterministic node '{var_name}': {e}"
                    )

    def transition(self, particle):
        """
        Perform a transition step.

        Parameters:
            particle: Particle
                Current particle to transition from.

        Returns:
            Particle: New particle after transition.
        """
        # Create new particle
        new_particle = particle.copy()

        # Get given values that should be fixed
        given_values = self.query.get_given_values()

        # Set fixed values from givens (only if not callable)
        for var_name, value in given_values.items():
            if not callable(value):
                new_particle.set_value(var_name, value)

        self._transition_non_deterministic(particle, new_particle)
        self._transition_deterministic_nodes(new_particle)

        # Check if new particle satisfies filters
        if not self._check_filters(new_particle):
            # If filters not satisfied, return original particle
            return particle

        return new_particle
