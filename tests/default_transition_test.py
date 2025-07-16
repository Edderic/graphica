import numpy as np
import pytest
from ..graphica.ds import BayesianNetwork as BN
from ..graphica.random.normal import Normal
from ..graphica.random.gamma import Gamma
from ..graphica.random.deterministic import Deterministic
from ..graphica.random.logistic import Logistic
from ..graphica.random.binomial import Binomial
from ..graphica.query import Query
from ..graphica.inference.default_transition import DefaultTransition
from ..graphica.particles.particle import Particle


def test_default_transition_basic():
    """Test basic functionality of DefaultTransition."""
    np.random.seed(42)

    # Create a simple Bayesian network
    bn = BN()

    # Add a Normal prior
    normal_prior = Normal(name='normal_param', mean=0, std=1)
    bn.add_node(normal_prior)

    # Add a Gamma prior
    gamma_prior = Gamma(name='gamma_param', shape=2, rate=1)
    bn.add_node(gamma_prior)

    # Add a Deterministic node
    def sum_func(normal_param, gamma_param):
        return normal_param + gamma_param

    det_node = Deterministic(name='deterministic_node', callable_func=sum_func,
                            normal_param=normal_prior, gamma_param=gamma_prior)
    bn.add_node(det_node)

    # Add a Logistic node
    def identity_func(deterministic_node):
        return deterministic_node

    logistic_node = Logistic(name='logistic_node', callable_func=identity_func,
                            deterministic_node=det_node)
    bn.add_node(logistic_node)

    # Create a query
    query = Query(
        outcomes=['normal_param', 'gamma_param', 'deterministic_node', 'logistic_node']
    )

    # Create DefaultTransition
    transition = DefaultTransition(bayesian_network=bn, query=query)

    # Create initial particle
    initial_particle = Particle({
        'normal_param': 0.5,
        'gamma_param': 1.5,
        'deterministic_node': 2.0,
        'logistic_node': 0.88
    })

    # Test transition
    new_particle = transition.transition(initial_particle)

    # Check that we got a new particle
    assert new_particle is not initial_particle

    # Check that all variables are present
    assert new_particle.has_variable('normal_param')
    assert new_particle.has_variable('gamma_param')
    assert new_particle.has_variable('deterministic_node')
    assert new_particle.has_variable('logistic_node')

    # Check that values changed (they should be perturbed)
    assert new_particle.get_value('normal_param') != initial_particle.get_value('normal_param')
    assert new_particle.get_value('gamma_param') != initial_particle.get_value('gamma_param')

    # Check that Deterministic chain was updated
    # The deterministic_node should reflect the new sum
    expected_sum = new_particle.get_value('normal_param') + new_particle.get_value('gamma_param')
    assert abs(new_particle.get_value('deterministic_node') - expected_sum) < 1e-10

    # The logistic_node should reflect the logistic transformation
    expected_logistic = 1.0 / (1.0 + np.exp(-expected_sum))
    assert abs(new_particle.get_value('logistic_node') - expected_logistic) < 1e-10


def test_default_transition_with_fixed_givens():
    """Test DefaultTransition with fixed values in givens."""
    np.random.seed(42)

    # Create a simple Bayesian bayesian_network
    bn = BN()

    # Add a Normal prior
    normal_prior = Normal(name='normal_param', mean=0, std=1)
    bn.add_node(normal_prior)

    # Add a Gamma prior
    gamma_prior = Gamma(name='gamma_param', shape=2, rate=1)
    bn.add_node(gamma_prior)

    # Create a query with fixed gamma_param
    query = Query(
        outcomes=['normal_param', 'gamma_param'],
        givens=[{'gamma_param': 2.0}]
    )

    # Create DefaultTransition
    transition = DefaultTransition(bayesian_network=bn, query=query)

    # Create initial particle
    initial_particle = Particle({
        'normal_param': 0.5,
        'gamma_param': 1.5
    })

    # Test transition
    new_particle = transition.transition(initial_particle)

    # Check that gamma_param is fixed
    assert new_particle.get_value('gamma_param') == 2.0

    # Check that normal_param changed
    assert new_particle.get_value('normal_param') != initial_particle.get_value('normal_param')


def test_default_transition_with_filters():
    """Test DefaultTransition with filter constraints."""
    np.random.seed(42)

    # Create a simple Bayesian bayesian_network
    bn = BN()

    # Add a Normal prior
    normal_prior = Normal(name='normal_param', mean=0, std=1)
    bn.add_node(normal_prior)

    # Create a query with filter
    query = Query(
        outcomes=['normal_param'],
        givens=[{'normal_param': lambda particle: particle.get_value('normal_param') > 0}]
    )

    # Create DefaultTransition
    transition = DefaultTransition(bayesian_network=bn, query=query)

    # Create initial particle with negative value
    initial_particle = Particle({'normal_param': -1.0})

    # Test transition - should return original particle if filter not satisfied
    new_particle = transition.transition(initial_particle)

    # Since the initial value is negative and the filter requires positive,
    # the transition should return the original particle
    assert new_particle is initial_particle


def test_default_transition_validation():
    """Test that DefaultTransition validates query variables exist in bayesian_network."""
    bn = BN()

    # Add only one variable
    normal_prior = Normal(name='normal_param', mean=0, std=1)
    bn.add_node(normal_prior)

    # Try to create query with non-existent variable
    query = Query(
        outcomes=['normal_param', 'non_existent_var']
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Outcome variable 'non_existent_var' not found"):
        DefaultTransition(bayesian_network=bn, query=query)


if __name__ == "__main__":
    test_default_transition_basic()
    test_default_transition_with_fixed_givens()
    test_default_transition_with_filters()
    test_default_transition_validation()
    print("All tests passed!")
