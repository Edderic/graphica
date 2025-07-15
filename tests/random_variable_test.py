import pytest
import numpy as np

from ..graphica.ds import RandomVariable


class ConcreteRandomVariable(RandomVariable):
    """Concrete implementation of RandomVariable for testing."""
    
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    
    def pdf(self, x):
        """Gaussian PDF."""
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
    
    def logpdf(self, x):
        """Gaussian log PDF."""
        return np.log(self.pdf(x))
    
    def sample(self, size=None):
        """Sample from Gaussian distribution."""
        return np.random.normal(self.mean, self.std, size=size)
    
    def __repr__(self):
        return f"ConcreteRandomVariable(mean={self.mean}, std={self.std})"


def test_random_variable_abstract_methods():
    """Test that RandomVariable cannot be instantiated directly."""
    with pytest.raises(TypeError):
        RandomVariable()


def test_concrete_random_variable_instantiation():
    """Test that a concrete implementation can be instantiated."""
    rv = ConcreteRandomVariable(mean=5, std=2)
    assert isinstance(rv, RandomVariable)
    assert rv.mean == 5
    assert rv.std == 2


def test_pdf_method():
    """Test the pdf method."""
    rv = ConcreteRandomVariable(mean=0, std=1)
    x = np.array([0, 1, -1])
    pdf_values = rv.pdf(x)
    
    assert len(pdf_values) == 3
    assert all(pdf_values > 0)
    # The PDF at x=0 should be the highest for a standard normal
    assert pdf_values[0] > pdf_values[1]
    assert pdf_values[0] > pdf_values[2]


def test_logpdf_method():
    """Test the logpdf method."""
    rv = ConcreteRandomVariable(mean=0, std=1)
    x = np.array([0, 1, -1])
    logpdf_values = rv.logpdf(x)
    pdf_values = rv.pdf(x)
    
    # Check that logpdf is the log of pdf
    np.testing.assert_array_almost_equal(logpdf_values, np.log(pdf_values))


def test_sample_method():
    """Test the sample method."""
    rv = ConcreteRandomVariable(mean=5, std=2)
    
    # Test single sample
    sample = rv.sample()
    assert isinstance(sample, (int, float, np.number))
    
    # Test multiple samples
    samples = rv.sample(size=1000)
    assert len(samples) == 1000
    assert isinstance(samples, np.ndarray)
    
    # Test that samples are roughly centered around the mean
    assert abs(np.mean(samples) - 5) < 0.5
    assert abs(np.std(samples) - 2) < 0.5


def test_string_representation():
    """Test string representation methods."""
    rv = ConcreteRandomVariable(mean=3, std=1.5)
    
    # Test __repr__
    repr_str = repr(rv)
    assert "ConcreteRandomVariable" in repr_str
    assert "mean=3" in repr_str
    assert "std=1.5" in repr_str
    
    # Test __str__
    str_str = str(rv)
    assert str_str == repr_str


def test_invalid_concrete_implementation():
    """Test that incomplete implementations raise errors."""
    class IncompleteRandomVariable(RandomVariable):
        def pdf(self, x):
            return x
    
    # Should raise TypeError because logpdf and sample are not implemented
    with pytest.raises(TypeError):
        IncompleteRandomVariable()


def test_abstract_methods_are_abstract():
    """Test that the abstract methods are properly marked as abstract."""
    # Check that the methods are abstract
    assert hasattr(RandomVariable.pdf, '__isabstractmethod__')
    assert hasattr(RandomVariable.logpdf, '__isabstractmethod__')
    assert hasattr(RandomVariable.sample, '__isabstractmethod__')
    
    # Check that they are marked as abstract
    assert RandomVariable.pdf.__isabstractmethod__
    assert RandomVariable.logpdf.__isabstractmethod__
    assert RandomVariable.sample.__isabstractmethod__


def test_parent_relationships():
    """Test parent relationship functionality."""
    rv = ConcreteRandomVariable(mean=0, std=1)
    
    # Test that it has the required methods
    assert hasattr(rv, 'get_parents')
    assert hasattr(rv, 'set_parents')
    
    # Test parent relationships with list
    parent_rv = ConcreteRandomVariable(mean=5, std=2)
    parent_rv.name = 'parent'
    rv.set_parents([parent_rv])
    assert rv.get_parents() == {'parent': parent_rv}
    
    # Test parent relationships with dict
    rv2 = ConcreteRandomVariable(mean=0, std=1)
    rv2.set_parents({'parent': parent_rv})
    assert rv2.get_parents() == {'parent': parent_rv}
    
    # Test invalid input
    with pytest.raises(TypeError):
        rv.set_parents("invalid") 