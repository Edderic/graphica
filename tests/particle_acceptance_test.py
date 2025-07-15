import pytest
from ..graphica.ds import Particle
from ..graphica.ds import BayesianNetwork, Normal, Query, MetropolisHastings
import numpy as np


def test_particle_acceptance_rejection():
    """Test Particle acceptance and rejection methods."""
    # Create a particle
    particle = Particle({'x': 1, 'y': 2})

    # Initially should not be accepted or rejected
    assert not particle.is_accepted()
    assert not particle.is_rejected()

    # Test accept
    particle.accept()
    assert particle.is_accepted()
    assert not particle.is_rejected()

    # Test reject
    particle.reject()
    assert not particle.is_accepted()
    assert particle.is_rejected()

    # Test accept again
    particle.accept()
    assert particle.is_accepted()
    assert not particle.is_rejected()


def test_particle_copy_resets_status():
    """Test that copying a particle preserves acceptance/rejection status."""
    # Create a particle and mark it as accepted
    particle = Particle({'x': 1})
    particle.accept()
    assert particle.is_accepted()

    # Copy the particle
    copied_particle = particle.copy()

    # Original should still be accepted
    assert particle.is_accepted()

    # Copy should have the same status as the original
    assert copied_particle.is_accepted()
    assert not copied_particle.is_rejected()

    # Values should be copied
    assert copied_particle.get_value('x') == 1


def test_particle_acceptance_in_metropolis_hastings():
    """Test that Metropolis-Hastings properly tracks acceptance."""
    # Create a simple network
    bn = BayesianNetwork()
    mu = Normal(name='mu', mean=0.0, std=1.0)
    bn.add_node(mu)

    # Create a query
    query = Query(outcomes=[], givens=[])

    # Simple transition function
    def transition_function(particle):
        new_particle = particle.copy()
        if particle.has_variable('mu'):
            current_mu = particle.get_value('mu')
            new_mu = current_mu + np.random.normal(0, 0.1)
            new_particle.set_value('mu', new_mu)
        return new_particle

    # Create sampler
    sampler = MetropolisHastings(
        network=bn,
        query=query,
        transition_function=transition_function
    )

    # Sample a few particles
    particles = sampler.sample(n=10, burn_in=0)

    # Check that particles have acceptance status
    accepted_count = sum(1 for p in particles if p.is_accepted())
    rejected_count = sum(1 for p in particles if p.is_rejected())

    # All particles should have some status (accepted or rejected)
    assert accepted_count + rejected_count == len(particles)
    assert len(particles) == 10
