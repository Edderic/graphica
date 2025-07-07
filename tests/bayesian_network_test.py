import pytest
import pandas as pd
import numpy as np

from ..linx.data import ParquetData
from ..linx.ds import BayesianNetwork as BN, ConditionalProbabilityTable as CPT, Particle
from .conftest import clean_tmp, get_tmp_path


def test_simple_network_with_cpts():
    """Test a simple Bayesian network using CPTs as RandomVariables."""
    clean_tmp()

    # Create a simple network: X -> Y
    bayesian_network = BN()
    
    # Prior for X
    cpt_x = CPT(
        table=[
            {'X': 0, 'value': 0.4},
            {'X': 1, 'value': 0.6}
        ],
        outcomes=['X'],
        name='X'
    )
    bayesian_network.add_node(cpt_x)

    # Conditional probability for Y given X
    cpt_y = CPT(
        table=[
            {'X': 0, 'Y': 0, 'value': 0.8},
            {'X': 0, 'Y': 1, 'value': 0.2},
            {'X': 1, 'Y': 0, 'value': 0.3},
            {'X': 1, 'Y': 1, 'value': 0.7}
        ],
        outcomes=['Y'],
        givens=['X'],
        name='Y'
    )
    bayesian_network.add_node(cpt_y)

    # Test sampling
    particle = bayesian_network.sample()
    assert isinstance(particle, Particle)
    assert particle.has_variable('X')
    assert particle.has_variable('Y')
    assert set(particle.get_variables()) == {'X', 'Y'}

    # Test that values are valid
    x_val = particle.get_value('X')
    y_val = particle.get_value('Y')
    assert x_val in [0, 1]
    assert y_val in [0, 1]


def test_network_with_mixed_random_variables():
    """Test a network with both CPTs and other RandomVariables."""
    clean_tmp()

    from ..linx.random.normal import Normal
    from ..linx.random.uniform import Uniform

    bayesian_network = BN()

    # Add a Normal random variable
    normal_rv = Normal(name='X', mu=0, sigma=1)
    bayesian_network.add_node(normal_rv)

    # Add a Uniform random variable
    uniform_rv = Uniform(name='Y', low=0, high=1)
    bayesian_network.add_node(uniform_rv)

    # Add a CPT that depends on both
    cpt_z = CPT(
        table=[
            {'X': 0, 'Y': 0, 'Z': 0, 'value': 0.5},
            {'X': 0, 'Y': 0, 'Z': 1, 'value': 0.5},
            {'X': 0, 'Y': 1, 'Z': 0, 'value': 0.3},
            {'X': 0, 'Y': 1, 'Z': 1, 'value': 0.7},
            {'X': 1, 'Y': 0, 'Z': 0, 'value': 0.7},
            {'X': 1, 'Y': 0, 'Z': 1, 'value': 0.3},
            {'X': 1, 'Y': 1, 'Z': 0, 'value': 0.2},
            {'X': 1, 'Y': 1, 'Z': 1, 'value': 0.8},
        ],
        outcomes=['Z'],
        givens=['X', 'Y'],
        name='Z'
    )
    bayesian_network.add_node(cpt_z)

    # Test sampling
    particle = bayesian_network.sample()
    assert isinstance(particle, Particle)
    assert set(particle.get_variables()) == {'X', 'Y', 'Z'}

    # Test that continuous variables have reasonable values
    x_val = particle.get_value('X')
    y_val = particle.get_value('Y')
    z_val = particle.get_value('Z')
    
    assert isinstance(x_val, (int, float))
    assert isinstance(y_val, (int, float))
    assert z_val in [0, 1]
    assert 0 <= y_val <= 1


def test_get_random_variables():
    """Test getting all random variables from the network."""
    clean_tmp()

    bayesian_network = BN()
    
    # Add some random variables
    cpt_x = CPT(
        table=[{'X': 0, 'value': 0.5}, {'X': 1, 'value': 0.5}],
        outcomes=['X'],
        name='X'
    )
    bayesian_network.add_node(cpt_x)

    cpt_y = CPT(
        table=[
            {'X': 0, 'Y': 0, 'value': 0.8},
            {'X': 0, 'Y': 1, 'value': 0.2},
            {'X': 1, 'Y': 0, 'value': 0.3},
            {'X': 1, 'Y': 1, 'value': 0.7}
        ],
        outcomes=['Y'],
        givens=['X'],
        name='Y'
    )
    bayesian_network.add_node(cpt_y)

    random_vars = bayesian_network.get_random_variables()
    assert len(random_vars) == 2
    assert 'X' in random_vars
    assert 'Y' in random_vars
    assert isinstance(random_vars['X'], CPT)
    assert isinstance(random_vars['Y'], CPT)


def test_add_edge():
    """Test adding edges between nodes."""
    clean_tmp()

    bayesian_network = BN()
    
    # Add nodes
    cpt_x = CPT(
        table=[{'X': 0, 'value': 0.5}, {'X': 1, 'value': 0.5}],
        outcomes=['X'],
        name='X'
    )
    bayesian_network.add_node(cpt_x)

    cpt_y = CPT(
        table=[
            {'X': 0, 'Y': 0, 'value': 0.8},
            {'X': 0, 'Y': 1, 'value': 0.2},
            {'X': 1, 'Y': 0, 'value': 0.3},
            {'X': 1, 'Y': 1, 'value': 0.7}
        ],
        outcomes=['Y'],
        givens=['X'],
        name='Y'
    )
    bayesian_network.add_node(cpt_y)

    # Test that edges were automatically added based on parent relationships
    assert 'X' in bayesian_network.get_parents('Y')
    assert 'Y' in bayesian_network.get_children('X')

    # Test manual edge addition
    bayesian_network.add_edge('X', 'Y')
    # Should not create duplicate edges
    assert len(bayesian_network.get_parents('Y')) == 1


def test_particle_class():
    """Test the Particle class functionality."""
    # Test initialization
    particle = Particle()
    assert particle.get_all_values() == {}

    # Test setting and getting values
    particle.set_value('X', 1)
    particle.set_value('Y', 0)

    assert particle.get_value('X') == 1
    assert particle.get_value('Y') == 0
    assert particle.has_variable('X')
    assert particle.has_variable('Y')
    assert not particle.has_variable('Z')

    # Test getting all values
    all_values = particle.get_all_values()
    assert all_values == {'X': 1, 'Y': 0}

    # Test getting variables
    variables = particle.get_variables()
    assert set(variables) == {'X', 'Y'}

    # Test initialization with values
    particle2 = Particle({'A': 1, 'B': 2})
    assert particle2.get_value('A') == 1
    assert particle2.get_value('B') == 2

    # Test string representation
    assert 'Particle' in str(particle)
    assert 'X' in str(particle)
