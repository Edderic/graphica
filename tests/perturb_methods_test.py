import numpy as np
import pytest
from ..graphica.random.normal import Normal
from ..graphica.random.gamma import Gamma
from ..graphica.random.beta import Beta
from ..graphica.random.uniform import Uniform
from ..graphica.random.binomial import Binomial
from ..graphica.random.deterministic import Deterministic


def test_normal_perturb():
    """Test Normal distribution perturb method."""
    np.random.seed(42)
    normal = Normal(name='test_normal', mean=0, std=1)
    
    current_value = 1.0
    perturbed = normal.perturb(current_value, mean=0, std=0.1)
    
    # Should be close to current value with small noise
    assert abs(perturbed - current_value) < 1.0
    assert perturbed != current_value  # Should be different due to noise


def test_gamma_perturb():
    """Test Gamma distribution perturb method."""
    np.random.seed(42)
    gamma = Gamma(name='test_gamma', shape=2, scale=1)
    
    current_value = 2.0
    perturbed = gamma.perturb(current_value, exp=0.1)
    
    # Should be positive and close to current value
    assert perturbed > 0
    assert abs(perturbed - current_value) < 1.0
    assert perturbed != current_value


def test_beta_perturb():
    """Test Beta distribution perturb method."""
    np.random.seed(42)
    beta = Beta(name='test_beta', a=2, b=3)
    
    current_value = 0.5
    perturbed = beta.perturb(current_value, low=-0.1, high=0.1)
    
    # Should be in [0, 1] and close to current value
    assert 0 <= perturbed <= 1
    assert abs(perturbed - current_value) < 0.2
    assert perturbed != current_value


def test_uniform_perturb():
    """Test Uniform distribution perturb method."""
    np.random.seed(42)
    uniform = Uniform(name='test_uniform', low=0, high=1)
    
    current_value = 0.5
    perturbed = uniform.perturb(current_value, low=-0.1, high=0.1)
    
    # Should be close to current value
    assert abs(perturbed - current_value) < 0.2
    assert perturbed != current_value


def test_binomial_perturb():
    """Test Binomial distribution perturb method."""
    np.random.seed(42)
    binomial = Binomial(name='test_binomial', n=10, p=0.5)
    
    current_value = 0.5
    perturbed = binomial.perturb(current_value, low=-0.1, high=0.1)
    
    # Should be in [0, 1] and close to current value
    assert 0 <= perturbed <= 1
    assert abs(perturbed - current_value) < 0.2
    assert perturbed != current_value


def test_deterministic_perturb():
    """Test Deterministic distribution perturb method."""
    np.random.seed(42)
    
    def identity_func(x):
        return x
    
    deterministic = Deterministic(name='test_deterministic', callable_func=identity_func, x=1.0)
    
    current_value = 1.0
    perturbed = deterministic.perturb(current_value)
    
    # Should be close to current value with small noise
    assert abs(perturbed - current_value) < 0.1
    assert perturbed != current_value


def test_perturb_with_defaults():
    """Test that perturb methods work with default parameters."""
    np.random.seed(42)
    
    normal = Normal(name='test_normal', mean=0, std=1)
    gamma = Gamma(name='test_gamma', shape=2, scale=1)
    beta = Beta(name='test_beta', a=2, b=3)
    uniform = Uniform(name='test_uniform', low=0, high=1)
    binomial = Binomial(name='test_binomial', n=10, p=0.5)
    
    # Use different current values to avoid edge cases
    normal_perturbed = normal.perturb(1.0)
    gamma_perturbed = gamma.perturb(2.0)
    beta_perturbed = beta.perturb(0.5)  # Use 0.5 instead of 1.0 to avoid clipping
    uniform_perturbed = uniform.perturb(0.5)
    binomial_perturbed = binomial.perturb(0.5)
    
    # All should return different values from their inputs
    assert normal_perturbed != 1.0
    assert gamma_perturbed != 2.0
    assert beta_perturbed != 0.5
    assert uniform_perturbed != 0.5
    assert binomial_perturbed != 0.5


if __name__ == "__main__":
    test_normal_perturb()
    test_gamma_perturb()
    test_beta_perturb()
    test_uniform_perturb()
    test_binomial_perturb()
    test_deterministic_perturb()
    test_perturb_with_defaults()
    print("All perturb method tests passed!") 