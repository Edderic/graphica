"""
Example demonstrating Beta random variables in a Bayesian network.
"""

import numpy as np
from ..ds import BayesianNetwork as BN, Query
from ..random.beta import Beta
from ..random.binomial import Binomial
from ..inference.metropolis_hastings import MetropolisHastings


def main():
    """Main example function."""
    print("Beta Random Variable Example")
    print("=" * 40)

    # Create a Bayesian network with Beta random variables
    bayesian_network = BN()

    # Add a Beta random variable as a prior for a probability parameter
    # This represents our prior belief about a probability
    p_prior = Beta(name="p_prior", alpha=2.0, beta_param=3.0)
    bayesian_network.add_node(p_prior)

    # Add a Binomial random variable that depends on the probability
    # This represents the number of successes in 10 trials
    success_rv = Binomial(name="successes", n=10, p=0.5)
    bayesian_network.add_node(success_rv)

    # Add another Beta random variable with different parameters
    # This represents another probability parameter
    q_prior = Beta(name="q_prior", alpha=1.0, beta_param=1.0)  # Uniform prior
    bayesian_network.add_node(q_prior)

    print("Bayesian Network created with:")
    print(f"  - Beta RV 'p_prior': α={p_prior.alpha}, β={p_prior.beta_param}")
    print(f"  - Binomial RV 'successes': n={success_rv.n}, p={success_rv.p}")
    print(f"  - Beta RV 'q_prior': α={q_prior.alpha}, β={q_prior.beta_param}")

    # Sample from the network
    print("\nSampling from the network:")
    particle = bayesian_network.sample()
    print(f"  Sampled particle: {particle.get_all_values()}")

    # Test individual Beta random variables
    print("\nTesting individual Beta random variables:")

    # Test different parameter combinations
    test_cases = [
        (1.0, 1.0, "Uniform (α=1, β=1)"),
        (2.0, 2.0, "Symmetric (α=2, β=2)"),
        (2.0, 5.0, "Skewed toward 0 (α=2, β=5)"),
        (5.0, 2.0, "Skewed toward 1 (α=5, β=2)"),
        (10.0, 10.0, "Concentrated around 0.5 (α=10, β=10)"),
        (0.5, 0.5, "U-shaped (α=0.5, β=0.5)"),
    ]

    for alpha, beta_param, description in test_cases:
        rv = Beta(name="test", alpha=alpha, beta_param=beta_param)
        print(f"\n  {description}:")

        # Test PDF at key points
        x_test = 0.5
        pdf_val = rv.pdf(x_test)
        print(f"    f(0.5) = {pdf_val:.4f}")

        # Test CDF
        cdf_val = rv.cdf(x_test)
        print(f"    F(0.5) = {cdf_val:.4f}")

        # Test sampling and mean
        samples = rv.sample(size=1000)
        mean_samples = np.mean(samples)
        expected_mean = alpha / (alpha + beta_param)
        print(f"    Sample mean: {mean_samples:.3f} (expected: {expected_mean:.3f})")

        # Test variance
        var_samples = np.var(samples)
        expected_var = (alpha * beta_param) / (
            (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
        )
        print(f"    Sample variance: {var_samples:.4f} (expected: {expected_var:.4f})")

    # Test special cases
    print("\nTesting special cases:")

    # Uniform case
    rv_uniform = Beta(alpha=1.0, beta_param=1.0)
    print(f"  Uniform case: f(0.3) = {rv_uniform.pdf(0.3):.4f}")
    print(f"  Uniform case: f(0.7) = {rv_uniform.pdf(0.7):.4f}")

    # Symmetric case
    rv_sym = Beta(alpha=3.0, beta_param=3.0)
    print(f"  Symmetric case: f(0.3) = {rv_sym.pdf(0.3):.4f}")
    print(f"  Symmetric case: f(0.7) = {rv_sym.pdf(0.7):.4f}")

    # Test consistency
    print("\nTesting mathematical consistency:")
    rv_test = Beta(alpha=2.0, beta_param=3.0)

    # Test that logpdf is log of pdf
    x_test = 0.6
    logpdf_val = rv_test.logpdf(x_test)
    pdf_val = rv_test.pdf(x_test)
    print(f"  log(f(0.6)) = {logpdf_val:.4f}")
    print(f"  log(f(0.6)) from pdf = {np.log(pdf_val):.4f}")

    # Test that CDF is 0 at 0 and 1 at 1
    print(f"  F(0) = {rv_test.cdf(0.0):.4f}")
    print(f"  F(1) = {rv_test.cdf(1.0):.4f}")

    # Test conjugate prior relationship with Binomial
    print("\nTesting conjugate prior relationship:")
    print("  Beta(α, β) is conjugate prior for Binomial(n, p)")
    print("  Posterior after observing k successes: Beta(α + k, β + n - k)")

    # Demonstrate with a simple example
    prior_alpha, prior_beta = 2.0, 3.0
    n_trials, k_successes = 10, 7

    prior = Beta(alpha=prior_alpha, beta_param=prior_beta)
    posterior = Beta(
        alpha=prior_alpha + k_successes, beta_param=prior_beta + n_trials - k_successes
    )

    print(f"  Prior: Beta({prior_alpha}, {prior_beta})")
    print(f"  Data: {k_successes} successes in {n_trials} trials")
    print(f"  Posterior: Beta({posterior.alpha}, {posterior.beta_param})")

    # Compare means
    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    posterior_mean = posterior.alpha / (posterior.alpha + posterior.beta_param)
    data_mean = k_successes / n_trials

    print(f"  Prior mean: {prior_mean:.3f}")
    print(f"  Data mean: {data_mean:.3f}")
    print(f"  Posterior mean: {posterior_mean:.3f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
