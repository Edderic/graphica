"""
Example demonstrating Binomial random variables in a Bayesian network.
"""

import numpy as np
from ..ds import BayesianNetwork as BN
from ..random.binomial import Binomial
from ..random.normal import Normal


def main():
    """Main example function."""
    print("Binomial Random Variable Example")
    print("=" * 40)

    # Create a Bayesian network with Binomial random variables
    bayesian_network = BN()

    # Add a Normal random variable for the probability parameter
    p_rv = Normal(name="p", mean=0.5, std=0.1)
    bayesian_network.add_node(p_rv)

    # Add a Binomial random variable that depends on p
    # This represents the number of successes in 10 trials
    success_rv = Binomial(name="successes", n=10, p=0.5)
    bayesian_network.add_node(success_rv)

    # Add another Binomial with different parameters
    # This represents the number of heads in 20 coin flips
    heads_rv = Binomial(name="heads", n=20, p=0.5)
    bayesian_network.add_node(heads_rv)

    print("Bayesian Network created with:")
    print(f"  - Normal RV 'p': μ={p_rv.mean}, σ={p_rv.std}")
    print(f"  - Binomial RV 'successes': n={success_rv.n}, p={success_rv.p}")
    print(f"  - Binomial RV 'heads': n={heads_rv.n}, p={heads_rv.p}")

    # Sample from the network
    print("\nSampling from the network:")
    particle = bayesian_network.sample()
    print(f"  Sampled particle: {particle.get_all_values()}")

    # Test individual Binomial random variables
    print("\nTesting individual Binomial random variables:")

    # Test different parameter combinations
    test_cases = [
        (5, 0.3, "n=5, p=0.3"),
        (10, 0.5, "n=10, p=0.5"),
        (20, 0.7, "n=20, p=0.7"),
        (1, 0.8, "Bernoulli case (n=1, p=0.8)"),
    ]

    for n, p, description in test_cases:
        rv = Binomial(name="test", n=n, p=p)
        print(f"\n  {description}:")

        # Test PDF
        x_test = min(3, n)  # Test a reasonable value
        pdf_val = rv.pdf(x_test)
        print(f"    P(X={x_test}) = {pdf_val:.4f}")

        # Test CDF
        cdf_val = rv.cdf(x_test)
        print(f"    P(X≤{x_test}) = {cdf_val:.4f}")

        # Test sampling
        samples = rv.sample(size=1000)
        mean_samples = np.mean(samples)
        expected_mean = n * p
        print(f"    Sample mean: {mean_samples:.2f} (expected: {expected_mean:.2f})")

    # Test special cases
    print("\nTesting special cases:")

    # n=0 case
    rv_zero = Binomial(n=0, p=0.5)
    print(f"  n=0 case: P(X=0) = {rv_zero.pdf(0):.4f}")

    # p=0 case
    rv_p0 = Binomial(n=10, p=0.0)
    print(f"  p=0 case: P(X=0) = {rv_p0.pdf(0):.4f}")

    # p=1 case
    rv_p1 = Binomial(n=10, p=1.0)
    print(f"  p=1 case: P(X=10) = {rv_p1.pdf(10):.4f}")

    # Test consistency
    print("\nTesting mathematical consistency:")
    rv_test = Binomial(n=8, p=0.4)

    # Test that PMF sums to 1
    total_prob = sum(rv_test.pdf(i) for i in range(rv_test.n + 1))
    print(f"  Sum of PMF values: {total_prob:.6f} (should be 1.0)")

    # Test that logpdf is log of pdf
    x_test = 3
    logpdf_val = rv_test.logpdf(x_test)
    pdf_val = rv_test.pdf(x_test)
    print(f"  log(P(X={x_test})) = {logpdf_val:.4f}")
    print(f"  log(P(X={x_test})) from pdf = {np.log(pdf_val):.4f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
