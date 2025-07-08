"""
Tests for Beta random variable
"""
import pytest
import numpy as np
from scipy.stats import beta

from ..linx.random.beta import Beta


def test_beta_initialization():
    """Test basic initialization of Beta random variable."""
    # Test with default parameters
    rv = Beta()
    assert rv.a == 1.0
    assert rv.b == 1.0
    assert rv.name is None

    # Test with custom parameters
    rv = Beta(name='X', a=2.0, b=3.0)
    assert rv.name == 'X'
    assert rv.a == 2.0
    assert rv.b == 3.0


def test_beter_validation():
    """Test parameter validation in Beta initialization."""
    # Test invalid a (non-positive)
    with pytest.raises(ValueError, match="a must be positive"):
        Beta(a=0.0, b=1.0)

    with pytest.raises(ValueError, match="a must be positive"):
        Beta(a=-1.0, b=1.0)

    # Test invalid beta (non-positive)
    with pytest.raises(ValueError, match="beta must be positive"):
        Beta(a=1.0, b=0.0)

    with pytest.raises(ValueError, match="beta must be positive"):
        Beta(a=1.0, b=-1.0)


def test_beta_pdf():
    """Test probability density function."""
    rv = Beta(a=2.0, b=3.0)

    # Test single values
    assert rv.pdf(0.0) == pytest.approx(beta.pdf(0.0, 2.0, 3.0))
    assert rv.pdf(0.5) == pytest.approx(beta.pdf(0.5, 2.0, 3.0))
    assert rv.pdf(1.0) == pytest.approx(beta.pdf(1.0, 2.0, 3.0))

    # Test array input
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = beta.pdf(x, 2.0, 3.0)
    result = rv.pdf(x)
    np.testing.assert_array_almost_equal(result, expected)

    # Test values outside support
    assert rv.pdf(-0.1) == 0.0
    assert rv.pdf(1.1) == 0.0


def test_beta_logpdf():
    """Test log probability density function."""
    rv = Beta(a=2.0, b=3.0)

    # Test single values
    assert rv.logpdf(0.5) == pytest.approx(beta.logpdf(0.5, 2.0, 3.0))
    assert rv.logpdf(0.25) == pytest.approx(beta.logpdf(0.25, 2.0, 3.0))

    # Test array input
    x = np.array([0.25, 0.5, 0.75])
    expected = beta.logpdf(x, 2.0, 3.0)
    result = rv.logpdf(x)
    np.testing.assert_array_almost_equal(result, expected)

    # Test values outside support
    assert rv.logpdf(-0.1) == -np.inf
    assert rv.logpdf(1.1) == -np.inf


def test_beta_sample():
    """Test sampling from beta distribution."""
    rv = Beta(a=2.0, b=3.0)

    # Test single sample
    sample = rv.sample()
    assert isinstance(sample, (float, np.floating))
    assert 0.0 <= sample <= 1.0

    # Test multiple samples
    samples = rv.sample(size=1000)
    assert len(samples) == 1000
    assert all(0.0 <= s <= 1.0 for s in samples)

    # Test array shape
    samples_2d = rv.sample(size=(10, 10))
    assert samples_2d.shape == (10, 10)
    assert all(0.0 <= s <= 1.0 for s in samples_2d.flatten())


def test_beta_cdf():
    """Test cumulative distribution function."""
    rv = Beta(a=2.0, b=3.0)

    # Test single values
    assert rv.cdf(0.0) == pytest.approx(beta.cdf(0.0, 2.0, 3.0))
    assert rv.cdf(0.5) == pytest.approx(beta.cdf(0.5, 2.0, 3.0))
    assert rv.cdf(1.0) == pytest.approx(beta.cdf(1.0, 2.0, 3.0))

    # Test array input
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = beta.cdf(x, 2.0, 3.0)
    result = rv.cdf(x)
    np.testing.assert_array_almost_equal(result, expected)

    # Test values outside support
    assert rv.cdf(-0.1) == 0.0
    assert rv.cdf(1.1) == 1.0


def test_beta_special_cases():
    """Test special cases of beta distribution."""
    # Test uniform case (a=1, beta=1)
    rv_uniform = Beta(a=1.0, b=1.0)
    assert rv_uniform.pdf(0.5) == pytest.approx(1.0)
    assert rv_uniform.pdf(0.25) == pytest.approx(1.0)

    # Test symmetric case (a=beta)
    rv_symmetric = Beta(a=2.0, b=2.0)
    assert rv_symmetric.pdf(0.5) == pytest.approx(rv_symmetric.pdf(0.5))
    assert rv_symmetric.pdf(0.25) == pytest.approx(rv_symmetric.pdf(0.75))

    # Test large parameters
    rv_large = Beta(a=10.0, b=10.0)
    # Should be concentrated around 0.5
    assert rv_large.pdf(0.5) > rv_large.pdf(0.1)
    assert rv_large.pdf(0.5) > rv_large.pdf(0.9)


def test_beta_conjugate_prior():
    """Test that Beta works as conjugate prior for Binomial."""
    # This is a common use case: Beta as prior for probability parameter
    rv = Beta(a=2.0, b=3.0)

    # Test that it's a proper probability distribution
    # The mean should be a/(a + beta)
    expected_mean = 2.0 / (2.0 + 3.0)  # 0.4

    # Sample to estimate mean
    samples = rv.sample(size=10000)
    sample_mean = np.mean(samples)
    assert abs(sample_mean - expected_mean) < 0.05

    # Test that values are in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in samples)


def test_beta_string_representation():
    """Test string representation of Beta random variable."""
    # Without name
    rv = Beta(a=2.0, b=3.0)
    assert str(rv) == "Beta(a=2.0, beta=3.0)"
    assert repr(rv) == "Beta(a=2.0, beta=3.0)"

    # With name
    rv = Beta(name='X', a=1.5, b=2.5)
    assert str(rv) == "Beta(name='X', a=1.5, beta=2.5)"
    assert repr(rv) == "Beta(name='X', a=1.5, beta=2.5)"


def test_beta_parent_relationships():
    """Test that Beta works with parent relationships."""
    rv = Beta(name='X', a=2.0, b=3.0)

    # Test that it has the required methods
    assert hasattr(rv, 'get_parents')
    assert hasattr(rv, 'set_parents')

    # Test parent relationships
    parent_rv = Beta(name='Y', a=1.0, b=1.0)
    rv.set_parents([parent_rv])
    assert rv.get_parents() == {'Y': parent_rv}


def test_beta_consistency():
    """Test consistency between pdf, logpdf, and cdf."""
    rv = Beta(a=3.0, b=2.0)

    # Test that logpdf is log of pdf
    x = 0.6
    assert rv.logpdf(x) == pytest.approx(np.log(rv.pdf(x)))

    # Test that cdf integrates pdf
    # For beta distribution, we can test at specific points
    assert rv.cdf(0.0) == 0.0
    assert rv.cdf(1.0) == 1.0

    # Test that pdf integrates to 1 (approximately)
    x_vals = np.linspace(0, 1, 1000)
    pdf_vals = rv.pdf(x_vals)
    integral = np.trapezoid(pdf_vals, x_vals)
    assert integral == pytest.approx(1.0, abs=0.01)


@pytest.mark.f
def test_beta_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test very small parameters
    rv_small = Beta(a=0.1, b=0.1)
    assert rv_small.pdf(0.5) > 0
    assert rv_small.pdf(0.0) == np.inf  # Beta(0.1, 0.1) has infinite density at boundaries
    assert rv_small.pdf(1.0) == np.inf

    # Test very large parameters
    rv_large = Beta(a=100.0, b=100.0)
    # Should be very concentrated around 0.5
    assert rv_large.pdf(0.5) > rv_large.pdf(0.4)
    assert rv_large.pdf(0.5) > rv_large.pdf(0.6)

    # Test asymmetric parameters
    rv_asym = Beta(a=1.0, b=5.0)
    # Should be skewed toward 0
    assert rv_asym.pdf(0.1) > rv_asym.pdf(0.9)
