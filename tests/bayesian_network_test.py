import pytest
import pandas as pd
import numpy as np

from ..linx.data import ParquetData
from ..linx.ds import BayesianNetwork as BN, ConditionalProbabilityTable as CPT, Particle
from .conftest import clean_tmp, get_tmp_path


def test_find_cpt_for_node():
    clean_tmp()

    bayesian_network = BN()
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        ParquetData(df_2, storage_folder=get_tmp_path()),
        outcomes=['A'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        ParquetData(df_3, storage_folder=get_tmp_path()),
        outcomes=['X'],
    )

    bayesian_network.add_node(cpt_3)

    assert bayesian_network.find_cpt_for_node('X') == cpt_3
    assert bayesian_network.find_cpt_for_node('A') == cpt_2
    assert bayesian_network.find_cpt_for_node('Y') == cpt_1


def test_to_markov_network():
    clean_tmp()

    bayesian_network = BN()
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])

    cpt_1 = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_1)

    df_2 = pd.DataFrame([
        {'X': 0, 'A': 0, 'value': 0.4},
        {'X': 0, 'A': 1, 'value': 0.6},
        {'X': 1, 'A': 0, 'value': 0.7},
        {'X': 1, 'A': 1, 'value': 0.3},
    ])

    cpt_2 = CPT(
        ParquetData(df_2, storage_folder=get_tmp_path()),
        outcomes=['A'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_2)

    df_3 = pd.DataFrame([
        {'X': 0, 'value': 0.2},
        {'X': 1, 'value': 0.8}
    ])

    cpt_3 = CPT(
        ParquetData(df_3, storage_folder=get_tmp_path()),
        outcomes=['X'],
    )

    bayesian_network.add_node(cpt_3)

    markov_network = bayesian_network.to_markov_network()
    factors = markov_network.get_factors()
    assert len(factors) == 3

def test_sample_simple_network():
    """Test sampling from a simple Bayesian Network with one variable."""
    clean_tmp()

    # Create a simple network with one variable X
    bayesian_network = BN()
    df = pd.DataFrame([
        {'X': 0, 'value': 0.3},
        {'X': 1, 'value': 0.7}
    ])

    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['X']
    )

    bayesian_network.add_node(cpt)

    # Sample multiple times and check that we get both values
    samples = []
    for _ in range(1000):
        particle = bayesian_network.sample()
        samples.append(particle.get_value('X'))

    mean = np.array(samples).mean()

    # Check that we get both values (0 and 1)
    unique_values = set(samples)
    assert 0 in unique_values
    assert 1 in unique_values

    # Check that the mean is within acceptable tolerance of 0.7
    assert abs(mean - 0.7) < 0.05, f"Expected mean close to 0.7, got {mean}"

    # Check that the particle has the correct structure
    particle = bayesian_network.sample()
    assert isinstance(particle, Particle)
    assert particle.has_variable('X')
    assert particle.get_variables() == ['X']


@pytest.mark.f
def test_sample_chain_network():
    """Test sampling from a chain network: X -> Y."""
    clean_tmp()

    # Create a chain network: X -> Y
    bayesian_network = BN()

    # Prior for X
    df_x = pd.DataFrame([
        {'X': 0, 'value': 0.4},
        {'X': 1, 'value': 0.6}
    ])

    cpt_x = CPT(
        ParquetData(df_x, storage_folder=get_tmp_path()),
        outcomes=['X']
    )

    bayesian_network.add_node(cpt_x)

    # Conditional probability for Y given X
    df_y = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.8},
        {'X': 0, 'Y': 1, 'value': 0.2},
        {'X': 1, 'Y': 0, 'value': 0.3},
        {'X': 1, 'Y': 1, 'value': 0.7}
    ])

    cpt_y = CPT(
        ParquetData(df_y, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )

    bayesian_network.add_edge(cpt_y)

    # Sample and verify structure
    particle = bayesian_network.sample()
    assert isinstance(particle, Particle)
    assert particle.has_variable('X')
    assert particle.has_variable('Y')
    assert set(particle.get_variables()) == {'X', 'Y'}

    # Sample multiple times and check that we get both values
    samples_y = []
    samples_x = []
    for _ in range(1000):
        particle = bayesian_network.sample()
        samples_y.append(particle.get_value('Y'))
        samples_x.append(particle.get_value('X'))


    # Check that the y mean is within acceptable tolerance
    expected_occurence_y = 0.4 * 0.8 + 0.6 * 0.3
    y_mean = np.array(samples_y).mean()
    assert abs(y_mean - expected_occurence_y) < 0.05, f"Expected mean close to {expected_occurence_y}, got {y_mean}"

    # Check that the x mean is within acceptable tolerance
    expected_occurence_x = 0.6
    x_mean = np.array(samples_x).mean()
    assert abs(x_mean - expected_occurence_x) < 0.05, f"Expected mean close to {expected_occurence_x}, got {x_mean}"

    # Check that values are valid
    x_val = particle.get_value('X')
    y_val = particle.get_value('Y')
    assert x_val in [0, 1]
    assert y_val in [0, 1]


def test_sample_complex_network():
    """Test sampling from a more complex network with multiple variables."""
    clean_tmp()

    # Create a network: A -> B, A -> C, B -> D
    bayesian_network = BN()

    # Prior for A
    df_a = pd.DataFrame([
        {'A': 0, 'value': 0.5},
        {'A': 1, 'value': 0.5}
    ])

    cpt_a = CPT(
        ParquetData(df_a, storage_folder=get_tmp_path()),
        outcomes=['A']
    )

    bayesian_network.add_node(cpt_a)

    # B depends on A
    df_b = pd.DataFrame([
        {'A': 0, 'B': 0, 'value': 0.7},
        {'A': 0, 'B': 1, 'value': 0.3},
        {'A': 1, 'B': 0, 'value': 0.2},
        {'A': 1, 'B': 1, 'value': 0.8}
    ])

    cpt_b = CPT(
        ParquetData(df_b, storage_folder=get_tmp_path()),
        outcomes=['B'],
        givens=['A']
    )

    bayesian_network.add_edge(cpt_b)

    # C depends on A
    df_c = pd.DataFrame([
        {'A': 0, 'C': 0, 'value': 0.6},
        {'A': 0, 'C': 1, 'value': 0.4},
        {'A': 1, 'C': 0, 'value': 0.1},
        {'A': 1, 'C': 1, 'value': 0.9}
    ])

    cpt_c = CPT(
        ParquetData(df_c, storage_folder=get_tmp_path()),
        outcomes=['C'],
        givens=['A']
    )

    bayesian_network.add_edge(cpt_c)

    # D depends on B
    df_d = pd.DataFrame([
        {'B': 0, 'D': 0, 'value': 0.9},
        {'B': 0, 'D': 1, 'value': 0.1},
        {'B': 1, 'D': 0, 'value': 0.4},
        {'B': 1, 'D': 1, 'value': 0.6}
    ])

    cpt_d = CPT(
        ParquetData(df_d, storage_folder=get_tmp_path()),
        outcomes=['D'],
        givens=['B']
    )

    bayesian_network.add_edge(cpt_d)

    # Sample and verify
    particle = bayesian_network.sample()
    assert isinstance(particle, Particle)
    assert set(particle.get_variables()) == {'A', 'B', 'C', 'D'}

    # Check that all values are valid
    for var in ['A', 'B', 'C', 'D']:
        val = particle.get_value(var)
        assert val in [0, 1]


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
