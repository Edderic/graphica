"""
Metropolis-Hastings sampler for Bayesian Networks with Random Variables.
"""
import numpy as np
from ..particles.particle import Particle
from tqdm import tqdm


class MetropolisHastings:
    """
    Metropolis-Hastings sampler for Bayesian Networks with Random Variables.

    Parameters:
        network: BayesianNetwork
            The Bayesian network to sample from.
        query: Query
            The query specifying what to condition on.
        transition_function: callable
            Function that takes a particle and returns a new proposed particle.
        initial_particle: Particle, optional
            Initial particle to start sampling from.
    """

    def __init__(self, network, query, transition_function, initial_particle=None):
        self.network = network
        self.query = query
        self.transition_function = transition_function
        self.initial_particle = initial_particle

    def sample(self, n=1000, burn_in=100):
        """
        Generate samples using Metropolis-Hastings.

        Parameters:
            n: int
                Number of samples to generate.
            burn_in: int
                Number of burn-in samples to discard.

        Returns:
            list[Particle]: List of sampled particles.
        """
        # Initialize particle
        if self.initial_particle is None:
            current_particle = self._initialize_particle()
        else:
            current_particle = self.initial_particle.copy()

        samples = []
        accepted = 0

        # Burn-in phase with progress bar
        for _ in tqdm(range(burn_in), desc="Burn-in", leave=False):
            current_particle = self._step(current_particle)

        # Sampling phase with progress bar
        for _ in tqdm(range(n), desc="Sampling"):
            current_particle = self._step(current_particle)
            samples.append(current_particle.copy())
            if current_particle.is_accepted():
                accepted += 1

        acceptance_rate = accepted / n
        print(f"Acceptance rate: {acceptance_rate:.3f}")

        return samples

    def _initialize_particle(self):
        """Initialize a particle by sampling from all random variables."""
        # Use BayesianNetwork.sample() which handles topological sorting and parent values
        particle = self.network.sample()

        # Set conditioned values (override sampled values)
        for var_name, value in self.query.get_given_values().items():
            particle.set_value(var_name, value)

        return particle

    def _step(self, current_particle):
        """Perform one Metropolis-Hastings step."""
        # Propose new particle
        proposed_particle = self.transition_function(current_particle)

        # Calculate acceptance ratio
        current_log_prob = self._log_probability(current_particle)
        proposed_log_prob = self._log_probability(proposed_particle)

        # Calculate proposal ratio (symmetric for most transition functions)
        log_alpha = proposed_log_prob - current_log_prob

        # Accept or reject
        alpha = np.exp(log_alpha)

        if alpha >= 1:
            proposed_particle.accept()
            return proposed_particle
        if np.random.random() < alpha:
            proposed_particle.accept()
            return proposed_particle
        else:
            current_particle.reject()
            return current_particle

    def _log_probability(self, particle):
        """Calculate log probability of a particle under the network."""
        log_prob = 0.0

        # Add log probabilities from all random variables
        for var_name, rv in self.network.random_variables.items():
            value = particle.get_value(var_name)

            parent_values = {}
            for parent_name, parent in rv.get_parents().items():
                parent_values[parent_name] = particle.get_value(parent.name)

            log_prob += rv.logpdf(value, **parent_values)

        return log_prob

    def _is_conditioned(self, var_name):
        """Check if a variable is conditioned on in the query."""
        given_values = self.query.get_given_values()
        return var_name in given_values
