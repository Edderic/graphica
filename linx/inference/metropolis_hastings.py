"""
Metropolis-Hastings sampler for Bayesian Networks with Random Variables.
"""
import numpy as np
from ..particles.particle import Particle


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
        
        # Burn-in phase
        for _ in range(burn_in):
            current_particle = self._step(current_particle)
        
        # Sampling phase
        for _ in range(n):
            current_particle = self._step(current_particle)
            samples.append(current_particle.copy())
        
        acceptance_rate = accepted / (burn_in + n)
        print(f"Acceptance rate: {acceptance_rate:.3f}")
        
        return samples
    
    def _initialize_particle(self):
        """Initialize a particle by sampling from all random variables."""
        particle = Particle()
        
        # Sample from all random variables in the network
        for var_name, rv in self.network.random_variables.items():
            if not self._is_conditioned(var_name):
                value = rv.sample()
                particle.set_value(var_name, value)
        
        # Set conditioned values
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
        if np.log(np.random.random()) < log_alpha:
            return proposed_particle
        else:
            return current_particle
    
    def _log_probability(self, particle):
        """Calculate log probability of a particle under the network."""
        log_prob = 0.0
        
        # Add log probabilities from all random variables
        for var_name, rv in self.network.random_variables.items():
            if particle.has_variable(var_name):
                value = particle.get_value(var_name)
                
                # For conditional random variables, we need parent values
                if hasattr(rv, 'get_parents') and rv.get_parents():
                    parent_values = {}
                    for parent in rv.get_parents():
                        if particle.has_variable(parent):
                            parent_values[parent] = particle.get_value(parent)
                        else:
                            # If parent not available, skip this variable
                            continue
                    log_prob += rv.logpdf(value, **parent_values)
                else:
                    log_prob += rv.logpdf(value)
        
        return log_prob
    
    def _is_conditioned(self, var_name):
        """Check if a variable is conditioned on in the query."""
        given_values = self.query.get_given_values()
        return var_name in given_values 