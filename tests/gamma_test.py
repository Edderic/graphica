import pytest
import numpy as np
from scipy import stats

from ..linx.ds import Gamma


def test_gamma_initialization():
    """Test gamma distribution initialization."""
    # Test default parameters
    gamma = Gamma(shape=2.0)
    assert gamma.shape == 2.0
    assert gamma.scale == 1.0
    
    # Test custom parameters
    gamma = Gamma(shape=3.0, scale=2.0)
    assert gamma.shape == 3.0
    assert gamma.scale == 2.0
    
    # Test invalid shape parameter
    with pytest.raises(ValueError, match="Shape parameter must be positive"):
        Gamma(shape=0)
    
    with pytest.raises(ValueError, match="Shape parameter must be positive"):
        Gamma(shape=-1)
    
    # Test invalid scale parameter
    with pytest.raises(ValueError, match="Scale parameter must be positive"):
        Gamma(shape=1.0, scale=0)
    
    with pytest.raises(ValueError, match="Scale parameter must be positive"):
        Gamma(shape=1.0, scale=-1)


def test_gamma_pdf():
    """Test the probability density function."""
    # Test with shape=2, scale=1
    gamma = Gamma(shape=2.0, scale=1.0)
    x = np.array([0.5, 1.0, 2.0, 3.0])
    pdf_values = gamma.pdf(x)
    
    # Compare with scipy's gamma distribution
    scipy_gamma = stats.gamma(a=2.0, scale=1.0)
    scipy_pdf = scipy_gamma.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, scipy_pdf)
    
    # Test that PDF is zero for negative values
    x_neg = np.array([-1.0, -0.5])
    pdf_neg = gamma.pdf(x_neg)
    np.testing.assert_array_almost_equal(pdf_neg, np.zeros_like(x_neg))


def test_gamma_logpdf():
    """Test the log probability density function."""
    gamma = Gamma(shape=2.0, scale=1.0)
    x = np.array([0.5, 1.0, 2.0, 3.0])
    logpdf_values = gamma.logpdf(x)
    pdf_values = gamma.pdf(x)
    
    # Check that logpdf is the log of pdf
    np.testing.assert_array_almost_equal(logpdf_values, np.log(pdf_values))
    
    # Compare with scipy
    scipy_gamma = stats.gamma(a=2.0, scale=1.0)
    scipy_logpdf = scipy_gamma.logpdf(x)
    np.testing.assert_array_almost_equal(logpdf_values, scipy_logpdf)


def test_gamma_sample():
    """Test sampling from gamma distribution."""
    gamma = Gamma(shape=2.0, scale=3.0)
    
    # Test single sample
    sample = gamma.sample()
    assert isinstance(sample, (int, float, np.number))
    assert sample > 0  # Gamma samples are always positive
    
    # Test multiple samples
    samples = gamma.sample(size=1000)
    assert len(samples) == 1000
    assert isinstance(samples, np.ndarray)
    assert all(samples > 0)
    
    # Test that samples have expected mean and variance
    # For Gamma(shape, scale): mean = shape * scale, var = shape * scale^2
    expected_mean = 2.0 * 3.0
    expected_var = 2.0 * 3.0 ** 2
    assert abs(np.mean(samples) - expected_mean) < 1.0
    assert abs(np.var(samples) - expected_var) < 5.0
    
    # Test 2D sampling
    samples_2d = gamma.sample(size=(10, 10))
    assert samples_2d.shape == (10, 10)
    assert all(samples_2d.flatten() > 0)


def test_gamma_cdf():
    """Test the cumulative distribution function."""
    gamma = Gamma(shape=2.0, scale=1.0)
    x = np.array([0.5, 1.0, 2.0, 3.0])
    cdf_values = gamma.cdf(x)
    
    # Compare with scipy
    scipy_gamma = stats.gamma(a=2.0, scale=1.0)
    scipy_cdf = scipy_gamma.cdf(x)
    
    np.testing.assert_array_almost_equal(cdf_values, scipy_cdf)
    
    # Test CDF properties
    assert cdf_values[0] < cdf_values[1] < cdf_values[2] < cdf_values[3]  # Monotonic
    assert cdf_values[0] >= 0  # CDF starts at 0
    assert cdf_values[-1] <= 1  # CDF approaches 1


def test_gamma_different_parameters():
    """Test gamma distribution with different parameters."""
    # Test with different shape and scale
    gamma = Gamma(shape=5.0, scale=2.0)
    x = np.array([2.0, 5.0, 10.0])
    pdf_values = gamma.pdf(x)
    
    # Compare with scipy
    scipy_gamma = stats.gamma(a=5.0, scale=2.0)
    scipy_pdf = scipy_gamma.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, scipy_pdf)
    
    # Test sampling
    samples = gamma.sample(size=1000)
    expected_mean = 5.0 * 2.0
    expected_var = 5.0 * 2.0 ** 2
    assert abs(np.mean(samples) - expected_mean) < 1.0
    assert abs(np.var(samples) - expected_var) < 5.0


def test_gamma_string_representation():
    """Test string representation methods."""
    gamma = Gamma(shape=3.5, scale=1.2)
    
    # Test __repr__
    repr_str = repr(gamma)
    assert "Gamma" in repr_str
    assert "shape=3.5" in repr_str
    assert "scale=1.2" in repr_str
    
    # Test __str__
    str_str = str(gamma)
    assert str_str == repr_str


def test_gamma_inheritance():
    """Test that Gamma properly inherits from RandomVariable."""
    gamma = Gamma(shape=2.0)
    assert isinstance(gamma, Gamma)
    assert hasattr(gamma, 'pdf')
    assert hasattr(gamma, 'logpdf')
    assert hasattr(gamma, 'sample')
    
    # Test that methods are callable
    assert callable(gamma.pdf)
    assert callable(gamma.logpdf)
    assert callable(gamma.sample)


def test_gamma_edge_cases():
    """Test edge cases for gamma distribution."""
    # Test with shape close to 0 (approaches exponential)
    gamma = Gamma(shape=0.1, scale=1.0)
    x = np.array([0.1, 1.0, 2.0])
    pdf_values = gamma.pdf(x)
    
    # PDF should be very high near 0, decreases rapidly
    assert pdf_values[0] > pdf_values[1]
    assert pdf_values[1] > pdf_values[2]
    
    # Test with large shape (approaches normal)
    gamma = Gamma(shape=10.0, scale=1.0)
    x = np.array([8.0, 10.0, 12.0])
    pdf_values = gamma.pdf(x)
    
    # PDF should be highest near the mean (shape * scale = 10)
    assert pdf_values[1] > pdf_values[0]
    assert pdf_values[1] > pdf_values[2]


def test_gamma_array_input():
    """Test that gamma methods handle array inputs correctly."""
    gamma = Gamma(shape=2.0, scale=1.0)
    
    # Test with different array shapes
    x_1d = np.array([0.5, 1.0, 2.0])
    x_2d = np.array([[0.5, 1.0], [2.0, 3.0]])
    
    pdf_1d = gamma.pdf(x_1d)
    pdf_2d = gamma.pdf(x_2d)
    
    assert pdf_1d.shape == (3,)
    assert pdf_2d.shape == (2, 2)
    
    # Test that values are correct
    assert all(pdf_1d > 0)
    assert all(pdf_2d.flatten() > 0)


def test_gamma_special_cases():
    """Test special cases of gamma distribution."""
    # Test exponential distribution (shape=1)
    gamma = Gamma(shape=1.0, scale=2.0)
    x = np.array([0.5, 1.0, 2.0])
    pdf_values = gamma.pdf(x)
    
    # Compare with exponential distribution
    scipy_exp = stats.expon(scale=2.0)
    exp_pdf = scipy_exp.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, exp_pdf)
    
    # Test chi-squared distribution (shape=k/2, scale=2)
    gamma = Gamma(shape=1.0, scale=2.0)  # chi-squared with 2 degrees of freedom
    x = np.array([0.5, 1.0, 2.0])
    pdf_values = gamma.pdf(x)
    
    scipy_chi2 = stats.chi2(df=2.0)
    chi2_pdf = scipy_chi2.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, chi2_pdf) 