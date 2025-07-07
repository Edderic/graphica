"""
Tests for Binomial random variable
"""
import pytest
import numpy as np
from scipy.stats import binom

from ..linx.random.binomial import Binomial


def test_binomial_initialization():
    """Test basic initialization of Binomial random variable."""
    # Test with default parameters
    rv = Binomial()
    assert rv.n == 1
    assert rv.p == 0.5
    assert rv.name is None
    
    # Test with custom parameters
    rv = Binomial(name='X', n=10, p=0.3)
    assert rv.name == 'X'
    assert rv.n == 10
    assert rv.p == 0.3


def test_binomial_parameter_validation():
    """Test parameter validation in Binomial initialization."""
    # Test invalid n (negative)
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        Binomial(n=-1, p=0.5)
    
    # Test invalid n (float)
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        Binomial(n=1.5, p=0.5)
    
    # Test invalid p (negative)
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        Binomial(n=10, p=-0.1)
    
    # Test invalid p (greater than 1)
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        Binomial(n=10, p=1.1)


def test_binomial_pdf():
    """Test probability mass function."""
    rv = Binomial(n=5, p=0.3)
    
    # Test single values
    assert rv.pdf(0) == pytest.approx(binom.pmf(0, 5, 0.3))
    assert rv.pdf(1) == pytest.approx(binom.pmf(1, 5, 0.3))
    assert rv.pdf(5) == pytest.approx(binom.pmf(5, 5, 0.3))
    
    # Test array input
    x = np.array([0, 1, 2, 3, 4, 5])
    expected = binom.pmf(x, 5, 0.3)
    result = rv.pdf(x)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test values outside support
    assert rv.pdf(-1) == 0.0
    assert rv.pdf(6) == 0.0


def test_binomial_logpdf():
    """Test log probability mass function."""
    rv = Binomial(n=5, p=0.3)
    
    # Test single values
    assert rv.logpdf(0) == pytest.approx(binom.logpmf(0, 5, 0.3))
    assert rv.logpdf(1) == pytest.approx(binom.logpmf(1, 5, 0.3))
    assert rv.logpdf(5) == pytest.approx(binom.logpmf(5, 5, 0.3))
    
    # Test array input
    x = np.array([0, 1, 2, 3, 4, 5])
    expected = binom.logpmf(x, 5, 0.3)
    result = rv.logpdf(x)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test values outside support
    assert rv.logpdf(-1) == -np.inf
    assert rv.logpdf(6) == -np.inf


def test_binomial_sample():
    """Test sampling from binomial distribution."""
    rv = Binomial(n=10, p=0.3)
    
    # Test single sample
    sample = rv.sample()
    assert isinstance(sample, (int, np.integer))
    assert 0 <= sample <= 10
    
    # Test multiple samples
    samples = rv.sample(size=1000)
    assert len(samples) == 1000
    assert all(0 <= s <= 10 for s in samples)
    
    # Test array shape
    samples_2d = rv.sample(size=(10, 10))
    assert samples_2d.shape == (10, 10)
    assert all(0 <= s <= 10 for s in samples_2d.flatten())


def test_binomial_cdf():
    """Test cumulative distribution function."""
    rv = Binomial(n=5, p=0.3)
    
    # Test single values
    assert rv.cdf(0) == pytest.approx(binom.cdf(0, 5, 0.3))
    assert rv.cdf(1) == pytest.approx(binom.cdf(1, 5, 0.3))
    assert rv.cdf(5) == pytest.approx(binom.cdf(5, 5, 0.3))
    
    # Test array input
    x = np.array([0, 1, 2, 3, 4, 5])
    expected = binom.cdf(x, 5, 0.3)
    result = rv.cdf(x)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test values outside support
    assert rv.cdf(-1) == 0.0
    assert rv.cdf(6) == 1.0


def test_binomial_special_cases():
    """Test special cases of binomial distribution."""
    # Test n=0 (degenerate case)
    rv = Binomial(n=0, p=0.5)
    assert rv.pdf(0) == 1.0
    assert rv.pdf(1) == 0.0
    assert rv.sample() == 0
    
    # Test p=0 (degenerate case)
    rv = Binomial(n=10, p=0.0)
    assert rv.pdf(0) == 1.0
    assert rv.pdf(1) == 0.0
    assert rv.sample() == 0
    
    # Test p=1 (degenerate case)
    rv = Binomial(n=10, p=1.0)
    assert rv.pdf(10) == 1.0
    assert rv.pdf(9) == 0.0
    assert rv.sample() == 10


def test_binomial_bernoulli():
    """Test that Binomial(n=1, p) behaves like Bernoulli(p)."""
    rv = Binomial(n=1, p=0.7)
    
    # Test PMF
    assert rv.pdf(0) == pytest.approx(0.3)
    assert rv.pdf(1) == pytest.approx(0.7)
    assert rv.pdf(2) == 0.0
    
    # Test CDF
    assert rv.cdf(0) == pytest.approx(0.3)
    assert rv.cdf(1) == pytest.approx(1.0)
    
    # Test sampling
    samples = rv.sample(size=1000)
    assert all(s in [0, 1] for s in samples)
    # Check that approximately 70% are 1s
    assert 0.65 <= np.mean(samples) <= 0.75


def test_binomial_string_representation():
    """Test string representation of Binomial random variable."""
    # Without name
    rv = Binomial(n=10, p=0.3)
    assert str(rv) == "Binomial(n=10, p=0.3)"
    assert repr(rv) == "Binomial(n=10, p=0.3)"
    
    # With name
    rv = Binomial(name='X', n=5, p=0.5)
    assert str(rv) == "Binomial(name='X', n=5, p=0.5)"
    assert repr(rv) == "Binomial(name='X', n=5, p=0.5)"


def test_binomial_parent_relationships():
    """Test that Binomial works with parent relationships."""
    rv = Binomial(name='X', n=10, p=0.3)
    
    # Test that it has the required methods
    assert hasattr(rv, 'get_parents')
    assert hasattr(rv, 'set_parents')
    
    # Test parent relationships
    parent_rv = Binomial(name='Y', n=5, p=0.5)
    rv.set_parents([parent_rv])
    assert rv.get_parents() == [parent_rv]


def test_binomial_consistency():
    """Test consistency between pdf, logpdf, and cdf."""
    rv = Binomial(n=8, p=0.4)
    
    # Test that logpdf is log of pdf
    x = 3
    assert rv.logpdf(x) == pytest.approx(np.log(rv.pdf(x)))
    
    # Test that cdf is sum of pdfs
    x = 4
    expected_cdf = sum(rv.pdf(i) for i in range(x + 1))
    assert rv.cdf(x) == pytest.approx(expected_cdf)
    
    # Test that pdf values sum to 1
    total_prob = sum(rv.pdf(i) for i in range(rv.n + 1))
    assert total_prob == pytest.approx(1.0) 