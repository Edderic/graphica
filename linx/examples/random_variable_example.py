"""
Example demonstrating the RandomVariable abstract class.
"""
import numpy as np
from ..ds import RandomVariable


class ExponentialRandomVariable(RandomVariable):
    """Exponential distribution implementation."""
    
    def __init__(self, rate=1.0):
        """
        Initialize exponential random variable.
        
        Parameters:
            rate: float
                Rate parameter (lambda) of the exponential distribution.
        """
        self.rate = rate
    
    def pdf(self, x):
        """Exponential PDF: f(x) = λ * exp(-λx) for x >= 0."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = x >= 0
        result[mask] = self.rate * np.exp(-self.rate * x[mask])
        return result
    
    def logpdf(self, x):
        """Exponential log PDF: log(f(x)) = log(λ) - λx for x >= 0."""
        x = np.asarray(x)
        result = np.full_like(x, -np.inf, dtype=float)
        mask = x >= 0
        result[mask] = np.log(self.rate) - self.rate * x[mask]
        return result
    
    def sample(self, size=None):
        """Sample from exponential distribution."""
        return np.random.exponential(1/self.rate, size=size)
    
    def __repr__(self):
        return f"ExponentialRandomVariable(rate={self.rate})"


class UniformRandomVariable(RandomVariable):
    """Uniform distribution implementation."""
    
    def __init__(self, low=0.0, high=1.0):
        """
        Initialize uniform random variable.
        
        Parameters:
            low: float
                Lower bound of the uniform distribution.
            high: float
                Upper bound of the uniform distribution.
        """
        self.low = low
        self.high = high
        self.width = high - low
    
    def pdf(self, x):
        """Uniform PDF: f(x) = 1/(high-low) for low <= x <= high."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= self.low) & (x <= self.high)
        result[mask] = 1.0 / self.width
        return result
    
    def logpdf(self, x):
        """Uniform log PDF: log(f(x)) = -log(high-low) for low <= x <= high."""
        x = np.asarray(x)
        result = np.full_like(x, -np.inf, dtype=float)
        mask = (x >= self.low) & (x <= self.high)
        result[mask] = -np.log(self.width)
        return result
    
    def sample(self, size=None):
        """Sample from uniform distribution."""
        return np.random.uniform(self.low, self.high, size=size)
    
    def __repr__(self):
        return f"UniformRandomVariable(low={self.low}, high={self.high})"


def run_random_variable_example():
    """Run the random variable example."""
    print("RandomVariable Abstract Class Example")
    print("=" * 50)
    
    # Create exponential random variable
    exp_rv = ExponentialRandomVariable(rate=2.0)
    print(f"Created: {exp_rv}")
    
    # Test PDF
    x = np.array([0, 0.5, 1.0, 2.0])
    pdf_values = exp_rv.pdf(x)
    print(f"\nPDF values at {x}: {pdf_values}")
    
    # Test log PDF
    logpdf_values = exp_rv.logpdf(x)
    print(f"Log PDF values at {x}: {logpdf_values}")
    
    # Test sampling
    samples = exp_rv.sample(size=5)
    print(f"5 samples: {samples}")
    
    print("\n" + "-" * 50)
    
    # Create uniform random variable
    uniform_rv = UniformRandomVariable(low=-1.0, high=1.0)
    print(f"Created: {uniform_rv}")
    
    # Test PDF
    x = np.array([-2, -1, 0, 1, 2])
    pdf_values = uniform_rv.pdf(x)
    print(f"\nPDF values at {x}: {pdf_values}")
    
    # Test log PDF
    logpdf_values = uniform_rv.logpdf(x)
    print(f"Log PDF values at {x}: {logpdf_values}")
    
    # Test sampling
    samples = uniform_rv.sample(size=5)
    print(f"5 samples: {samples}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    run_random_variable_example() 