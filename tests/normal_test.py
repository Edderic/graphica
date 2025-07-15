import pytest
import numpy as np
from scipy import stats

from ..graphica.ds import Normal


def test_normal_initialization():
    """Test normal distribution initialization."""
    # Test default parameters
    normal = Normal()
    assert normal.mean == 0.0
    assert normal.std == 1.0
    assert normal.var == 1.0
    
    # Test custom parameters
    normal = Normal(mean=5.0, std=2.0)
    assert normal.mean == 5.0
    assert normal.std == 2.0
    assert normal.var == 4.0
    
    # Test invalid standard deviation
    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        Normal(std=0)
    
    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        Normal(std=-1)


def test_normal_pdf():
    """Test the probability density function."""
    # Standard normal distribution
    normal = Normal(mean=0, std=1)
    x = np.array([-2, -1, 0, 1, 2])
    pdf_values = normal.pdf(x)
    
    # Compare with scipy's normal distribution
    scipy_normal = stats.norm(loc=0, scale=1)
    scipy_pdf = scipy_normal.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, scipy_pdf)
    
    # Test that PDF is symmetric around mean
    assert abs(pdf_values[1] - pdf_values[3]) < 1e-10  # f(-1) = f(1)
    assert abs(pdf_values[0] - pdf_values[4]) < 1e-10  # f(-2) = f(2)
    
    # Test that maximum is at mean
    assert pdf_values[2] > pdf_values[1]  # f(0) > f(1)
    assert pdf_values[2] > pdf_values[3]  # f(0) > f(-1)


def test_normal_logpdf():
    """Test the log probability density function."""
    normal = Normal(mean=0, std=1)
    x = np.array([-2, -1, 0, 1, 2])
    logpdf_values = normal.logpdf(x)
    pdf_values = normal.pdf(x)
    
    # Check that logpdf is the log of pdf
    np.testing.assert_array_almost_equal(logpdf_values, np.log(pdf_values))
    
    # Compare with scipy
    scipy_normal = stats.norm(loc=0, scale=1)
    scipy_logpdf = scipy_normal.logpdf(x)
    np.testing.assert_array_almost_equal(logpdf_values, scipy_logpdf)


def test_normal_sample():
    """Test sampling from normal distribution."""
    normal = Normal(mean=5, std=2)
    
    # Test single sample
    sample = normal.sample()
    assert isinstance(sample, (int, float, np.number))
    
    # Test multiple samples
    samples = normal.sample(size=1000)
    assert len(samples) == 1000
    assert isinstance(samples, np.ndarray)
    
    # Test that samples are roughly centered around the mean
    assert abs(np.mean(samples) - 5) < 0.5
    assert abs(np.std(samples) - 2) < 0.5
    
    # Test 2D sampling
    samples_2d = normal.sample(size=(10, 10))
    assert samples_2d.shape == (10, 10)


def test_normal_cdf():
    """Test the cumulative distribution function."""
    normal = Normal(mean=0, std=1)
    x = np.array([-2, -1, 0, 1, 2])
    cdf_values = normal.cdf(x)
    
    # Compare with scipy
    scipy_normal = stats.norm(loc=0, scale=1)
    scipy_cdf = scipy_normal.cdf(x)
    
    np.testing.assert_array_almost_equal(cdf_values, scipy_cdf, decimal=5)
    
    # Test CDF properties
    assert cdf_values[0] < cdf_values[1] < cdf_values[2] < cdf_values[3] < cdf_values[4]  # Monotonic
    assert abs(cdf_values[2] - 0.5) < 0.01  # CDF at mean should be 0.5


def test_normal_different_parameters():
    """Test normal distribution with different parameters."""
    # Test with different mean and std
    normal = Normal(mean=10, std=3)
    x = np.array([7, 10, 13])
    pdf_values = normal.pdf(x)
    
    # Compare with scipy
    scipy_normal = stats.norm(loc=10, scale=3)
    scipy_pdf = scipy_normal.pdf(x)
    
    np.testing.assert_array_almost_equal(pdf_values, scipy_pdf)
    
    # Test sampling
    samples = normal.sample(size=1000)
    assert abs(np.mean(samples) - 10) < 0.5
    assert abs(np.std(samples) - 3) < 0.5


def test_normal_string_representation():
    """Test string representation methods."""
    normal = Normal(mean=3.5, std=1.2)
    
    # Test __repr__
    repr_str = repr(normal)
    assert "Normal" in repr_str
    assert "mean=3.5" in repr_str
    assert "std=1.2" in repr_str
    
    # Test __str__
    str_str = str(normal)
    assert str_str == repr_str


def test_normal_inheritance():
    """Test that Normal properly inherits from RandomVariable."""
    normal = Normal()
    assert isinstance(normal, Normal)
    assert hasattr(normal, 'pdf')
    assert hasattr(normal, 'logpdf')
    assert hasattr(normal, 'sample')
    
    # Test that methods are callable
    assert callable(normal.pdf)
    assert callable(normal.logpdf)
    assert callable(normal.sample)


def test_normal_edge_cases():
    """Test edge cases for normal distribution."""
    # Test with very small standard deviation
    normal = Normal(mean=0, std=0.1)
    x = np.array([0, 0.1, -0.1])
    pdf_values = normal.pdf(x)
    
    # PDF at mean should be greater than at 1 std away
    assert pdf_values[0] > pdf_values[1]
    assert pdf_values[0] > pdf_values[2]
    # Theoretical ratio for normal: exp(0.5)
    expected_ratio = np.exp(0.5)
    actual_ratio_pos = pdf_values[0] / pdf_values[1]
    actual_ratio_neg = pdf_values[0] / pdf_values[2]
    assert abs(actual_ratio_pos - expected_ratio) < 1e-3
    assert abs(actual_ratio_neg - expected_ratio) < 1e-3
    
    # Test with very large standard deviation
    normal = Normal(mean=0, std=10)
    x = np.array([0, 1, -1])
    pdf_values = normal.pdf(x)
    
    # PDF should be more uniform (but not exactly equal due to normal distribution shape)
    assert abs(pdf_values[0] - pdf_values[1]) < 0.001  # Very close
    assert abs(pdf_values[0] - pdf_values[2]) < 0.001


def test_normal_array_input():
    """Test that normal methods handle array inputs correctly."""
    normal = Normal(mean=0, std=1)
    
    # Test with different array shapes
    x_1d = np.array([0, 1, 2])
    x_2d = np.array([[0, 1], [2, 3]])
    
    pdf_1d = normal.pdf(x_1d)
    pdf_2d = normal.pdf(x_2d)
    
    assert pdf_1d.shape == (3,)
    assert pdf_2d.shape == (2, 2)
    
    # Test that values are correct
    assert all(pdf_1d > 0)
    assert all(pdf_2d.flatten() > 0) 