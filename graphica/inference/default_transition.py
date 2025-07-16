"""
DefaultTransition class for Metropolis-Hastings sampling.
"""

import numpy as np
from ..query import Query
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
        gamma_args: dict, optional
            Arguments for Gamma perturbation (default: {'exp': 0.1}).
        normal_args: dict, optional
            Arguments for Normal perturbation (default: {'mean': 0, 'std': 1}).
        beta_args: dict, optional
            Arguments for Beta perturbation (default: {'alpha': 1, 'beta': 1}).
        uniform_args: dict, optional
            Arguments for Uniform perturbation (default: {'low': -0.1, 'high': 0.1}).
    """

    def __init__(
        self,
        bayesian_network,
        query,
        gamma_args=None,
        normal_args=None,
        beta_args=None,
        uniform_args=None,
    ):
        self.bayesian_network = bayesian_network
        self.query = query

        # Set default perturbation arguments
        self.gamma_args = gamma_args or {"exp": 0.1}
        self.normal_args = normal_args or {"mean": 0, "std": 1}
        self.beta_args = beta_args or {"alpha": 1, "beta": 1}
        self.uniform_args = uniform_args or {"low": -0.1, "high": 0.1}

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
        """Get all Deterministic nodes that depend on start_var (including start_var if it's Deterministic)."""
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

    def _perturb_value(self, var_name, current_value):
        """Perturb a value using the distribution's perturb method."""
        rv = self.bayesian_network.random_variables[var_name]

        # Get perturbation parameters based on distribution type
        if isinstance(rv, Normal):
            return rv.perturb(current_value, **self.normal_args)
        elif isinstance(rv, Gamma):
            return rv.perturb(current_value, **self.gamma_args)
        elif isinstance(rv, Beta):
            return rv.perturb(current_value, **self.beta_args)
        elif isinstance(rv, Uniform):
            return rv.perturb(current_value, **self.uniform_args)
        elif isinstance(rv, Binomial):
            return rv.perturb(current_value, **self.uniform_args)
        elif isinstance(rv, Deterministic):
            return rv.perturb(current_value)
        else:
            # Use the distribution's perturb method with default parameters
            return rv.perturb(current_value)

    def _update_deterministic_chain(self, particle, chain_vars):
        """Update all Deterministic nodes in the chain."""
        for var_name in chain_vars:
            if var_name in self.deterministic_nodes:
                rv = self.deterministic_nodes[var_name]

                # Gather required parameters for the Deterministic node
                import inspect

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
                    else:
                        # Get from particle
                        if not particle.has_variable(param_name):
                            raise ValueError(
                                f"Missing parameter '{param_name}' for Deterministic node '{var_name}'"
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
            new_value = self._perturb_value(var_name, current_value)
            new_particle.set_value(var_name, new_value)

            # Update Deterministic chain (deprecated: we'll do a full update after all transitions)
            # chain_vars = self._get_deterministic_chain(var_name)
            # self._update_deterministic_chain(new_particle, chain_vars)

        # After all transitions, update all Deterministic nodes in topological order
        # (to ensure all values are consistent with the current particle state)
        # Use the BayesianNetwork's topological sort
        if hasattr(self.bayesian_network, "topological_sort"):
            topo_order = self.bayesian_network.topological_sort()
        else:
            topo_order = self.bayesian_network.get_nodes()  # fallback
        for var_name in topo_order:
            if var_name in self.deterministic_nodes:
                rv = self.deterministic_nodes[var_name]
                import inspect

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
                    else:
                        if not new_particle.has_variable(param_name):
                            raise ValueError(
                                f"Missing parameter '{param_name}' for Deterministic node '{var_name}'"
                            )
                        params[param_name] = new_particle.get_value(param_name)
                try:
                    new_value = rv.sample(**params)
                    new_particle.set_value(var_name, new_value)
                except Exception as e:
                    raise ValueError(
                        f"Error sampling Deterministic node '{var_name}': {e}"
                    )

        # Check if new particle satisfies filters
        if not self._check_filters(new_particle):
            # If filters not satisfied, return original particle
            return particle

        return new_particle
