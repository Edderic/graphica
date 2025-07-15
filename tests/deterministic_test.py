import numpy as np
import pytest
import uuid
from ..graphica.random.normal import Normal
from ..graphica.random.deterministic import Deterministic
from ..graphica.random.logistic import Logistic


def test_deterministic_basic_functionality():
    """Test basic functionality of Deterministic class."""
    # Test with simple deterministic
    deterministic = Deterministic(lambda: 123)
    assert deterministic.logpdf(0) == 0
    assert deterministic.pdf(0) == 1
    assert deterministic.sample() == 123
    assert deterministic.get_parents() == {}


def test_deterministic_with_parameters():
    """Test Deterministic class with parameters as in the example."""
    beta_1 = Normal(mean=0, std=1)
    beta_2 = Normal(mean=1, std=3)
    value_1 = 1.5
    value_2 = 0.5

    deterministic = Deterministic(
        lambda beta_1, value_1, beta_2, value_2: beta_1 * value_1 + beta_2 * value_2,
        beta_1=beta_1,
        beta_2=beta_2,
        value_1=value_1,
        value_2=value_2
    )

    # Test logpdf and pdf
    assert deterministic.logpdf(0) == 0
    assert deterministic.pdf(0) == 1

    # Test sample with provided values
    result = deterministic.sample(beta_1=1, beta_2=2)
    expected = 1 * 1.5 + 2 * 0.5  # 1.5 + 1.0 = 2.5
    assert result == expected

    # Test get_parents
    parents = deterministic.get_parents()
    assert len(parents) == 2
    assert 'beta_1' in parents
    assert 'beta_2' in parents
    assert parents['beta_1'] == beta_1
    assert parents['beta_2'] == beta_2


def test_logistic_basic_functionality():
    """Test basic functionality of Logistic class."""
    # Test with simple constant
    logistic = Logistic(lambda: 0)
    assert logistic.logpdf(0) == 0
    assert logistic.pdf(0) == 1
    assert logistic.sample() == 0.5  # 1/(1+exp(0)) = 0.5


def test_logistic_with_parameters():
    """Test Logistic class with parameters as in the example."""
    beta_1 = Normal(0, 1)
    value_1 = 1.5
    beta_2 = Normal(0, 2)
    value_2 = -2

    logistic = Logistic(
        lambda beta_1, value_1, beta_2, value_2: beta_1 * value_1 + beta_2 * value_2,
        beta_1=beta_1,
        beta_2=beta_2,
        value_1=value_1,
        value_2=value_2
    )

    # Test logpdf and pdf
    assert logistic.logpdf(0) == 0
    assert logistic.pdf(0) == 1

    # Test get_parents
    parents = logistic.get_parents()
    assert len(parents) == 2
    assert 'beta_1' in parents
    assert 'beta_2' in parents
    assert parents['beta_1'] == beta_1
    assert parents['beta_2'] == beta_2

    # Test sample with provided values
    result = logistic.sample(beta_1=0.25, beta_2=1)
    # Expected: 1.0 / (1.0 + np.exp(-0.25 * 1.5 + 1 * -2))
    # = 1.0 / (1.0 + np.exp(-0.375 - 2))
    # = 1.0 / (1.0 + np.exp(-1.625))
    # ≈ 1.0 / (1.0 + 0.197) ≈ 1.0 / 1.197 ≈ 0.165
    expected = 1.0 / (1.0 + np.exp(-(0.25 * 1.5 + 1 * -2)))
    assert abs(result - expected) < 1e-10


def test_deterministic_error_handling():
    """Test error handling in Deterministic class."""
    deterministic = Deterministic(lambda x: x)
    
    # Test with missing parameter
    with pytest.raises(ValueError):
        deterministic.sample()  # Missing x parameter
    
    # Test with function that raises an error
    deterministic_bad = Deterministic(lambda x: x / 0)
    with pytest.raises(ValueError):
        deterministic_bad.sample(x=1)  # Division by zero


def test_logistic_error_handling():
    """Test error handling in Logistic class."""
    logistic = Logistic(lambda x: x)

    # Test with missing parameter
    with pytest.raises(ValueError):
        logistic.sample()  # Missing x parameter


def test_deterministic_size_parameter():
    """Test Deterministic class with size parameter."""
    deterministic = Deterministic(lambda: 123)
    
    # Test scalar size
    result = deterministic.sample(size=3)
    assert result.shape == (3,)
    assert np.all(result == 123)
    
    # Test tuple size
    result = deterministic.sample(size=(2, 3))
    assert result.shape == (2, 3)
    assert np.all(result == 123)


def test_logistic_size_parameter():
    """Test Logistic class with size parameter."""
    logistic = Logistic(lambda: 0)

    # Test scalar size
    result = logistic.sample(size=3)
    assert result.shape == (3,)
    assert np.all(result == 0.5)

    # Test tuple size
    result = logistic.sample(size=(2, 3))
    assert result.shape == (2, 3)
    assert np.all(result == 0.5)


def test_deterministic_array_output():
    """Test Deterministic class with array output."""
    deterministic = Deterministic(lambda: np.array([1, 2, 3]))
    
    result = deterministic.sample()
    assert np.array_equal(result, np.array([1, 2, 3]))
    
    # Test with size parameter
    result = deterministic.sample(size=2)
    assert result.shape == (2, 3)
    assert np.array_equal(result[0], np.array([1, 2, 3]))
    assert np.array_equal(result[1], np.array([1, 2, 3]))


def test_logistic_array_output():
    """Test Logistic class with array output."""
    logistic = Logistic(lambda: np.array([0, 1, -1]))
    
    result = logistic.sample()
    expected = 1.0 / (1.0 + np.exp(-np.array([0, 1, -1])))
    assert np.allclose(result, expected)


def test_uuid_generation():
    """Test that UUIDs are generated when no name is provided."""
    # Test Deterministic class
    deterministic = Deterministic(lambda: 123)
    assert deterministic.name is not None
    # Verify it's a valid UUID
    uuid.UUID(deterministic.name)
    
    # Test Logistic class
    logistic = Logistic(lambda: 0)
    assert logistic.name is not None
    # Verify it's a valid UUID
    uuid.UUID(logistic.name)
    
    # Test that different instances get different UUIDs
    deterministic2 = Deterministic(lambda: 456)
    assert deterministic.name != deterministic2.name
    
    logistic2 = Logistic(lambda: 1)
    assert logistic.name != logistic2.name


def test_name_parameter():
    """Test that explicit names are used when provided."""
    # Test Deterministic class
    deterministic = Deterministic(lambda: 123, name="test_deterministic")
    assert deterministic.name == "test_deterministic"
    
    # Test Logistic class
    logistic = Logistic(lambda: 0, name="test_logistic")
    assert logistic.name == "test_logistic"
