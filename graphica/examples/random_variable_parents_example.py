"""
Random Variable Parents Example

This example demonstrates how to create random variables with parent dependencies.
"""

from graphica.ds import Normal, Beta, Binomial, Gamma, Uniform


def main():
    print("=== Random Variable Parent Relationships and kwargs Override Example ===\n")

    # Example 1: Normal distribution with parent relationships
    print("1. Normal distribution with parent relationships:")

    # Create parent random variables
    mean_parent = Normal(name="mean_parent", mean=5.0, std=1.0)
    std_parent = Gamma(name="std_parent", shape=2.0, scale=1.0)

    # Create normal distribution with parents
    normal_rv = Normal(name="X", mean=mean_parent, std=std_parent)

    print(f"   Normal RV: {normal_rv}")
    print(f"   Parents: {normal_rv.get_parents()}")

    # Sample parent values and use them to override parameters
    parent_mean = mean_parent.sample()
    parent_std = std_parent.sample()

    print(f"   Sampled parent values: mean={parent_mean:.3f}, std={parent_std:.3f}")

    # Use kwargs to override parameters
    x = 6.0
    pdf_value = normal_rv.pdf(x, mean=parent_mean, std=parent_std)
    logpdf_value = normal_rv.logpdf(x, mean=parent_mean, std=parent_std)

    print(f"   PDF({x}) with overridden parameters: {pdf_value:.6f}")
    print(f"   LogPDF({x}) with overridden parameters: {logpdf_value:.6f}")
    print()

    # Example 2: Beta distribution with parent relationships
    print("2. Beta distribution with parent relationships:")

    # Create parent random variables for Beta parameters
    a_parent = Gamma(name="a_parent", shape=2.0, scale=1.0)
    b_parent = Gamma(name="b_parent", shape=3.0, scale=1.0)

    # Create beta distribution with parents
    beta_rv = Beta(name="theta", a=a_parent, b=b_parent)

    print(f"   Beta RV: {beta_rv}")
    print(f"   Parents: {beta_rv.get_parents()}")

    # Sample parent values
    parent_a = a_parent.sample()
    parent_b = b_parent.sample()

    print(f"   Sampled parent values: a={parent_a:.3f}, b={parent_b:.3f}")

    # Use kwargs to override parameters
    x = 0.7
    pdf_value = beta_rv.pdf(x, a=parent_a, b=parent_b)
    logpdf_value = beta_rv.logpdf(x, a=parent_a, b=parent_b)

    print(f"   PDF({x}) with overridden parameters: {pdf_value:.6f}")
    print(f"   LogPDF({x}) with overridden parameters: {logpdf_value:.6f}")
    print()

    # Example 3: Binomial distribution with parent relationships
    print("3. Binomial distribution with parent relationships:")

    # Create parent random variables for Binomial parameters
    n_parent = Uniform(name="n_parent", low=10, high=20)
    p_parent = Beta(name="p_parent", a=2.0, b=3.0)

    # Create binomial distribution with parents
    binomial_rv = Binomial(name="successes", n=n_parent, p=p_parent)

    print(f"   Binomial RV: {binomial_rv}")
    print(f"   Parents: {binomial_rv.get_parents()}")

    # Sample parent values
    parent_n = int(n_parent.sample())
    parent_p = p_parent.sample()

    print(f"   Sampled parent values: n={parent_n}, p={parent_p:.3f}")

    # Use kwargs to override parameters
    x = 5
    pmf_value = binomial_rv.pdf(x, n=parent_n, p=parent_p)
    logpmf_value = binomial_rv.logpdf(x, n=parent_n, p=parent_p)

    print(f"   PMF({x}) with overridden parameters: {pmf_value:.6f}")
    print(f"   LogPMF({x}) with overridden parameters: {logpmf_value:.6f}")
    print()

    # Example 4: Sampling with overridden parameters
    print("4. Sampling with overridden parameters:")

    # Create a simple normal distribution
    simple_normal = Normal(name="Y", mean=0.0, std=1.0)

    # Sample with default parameters
    default_sample = simple_normal.sample()
    print(f"   Default sample: {default_sample:.3f}")

    # Sample with overridden parameters
    overridden_sample = simple_normal.sample(mean=10.0, std=2.0)
    print(f"   Sample with mean=10, std=2: {overridden_sample:.3f}")

    # Sample multiple values with overridden parameters
    multiple_samples = simple_normal.sample(size=5, mean=5.0, std=1.5)
    print(f"   Multiple samples with mean=5, std=1.5: {multiple_samples}")
    print()

    # Example 5: CDF with overridden parameters
    print("5. CDF with overridden parameters:")

    gamma_rv = Gamma(name="Z", shape=2.0, scale=1.0)
    x_values = [0.5, 1.0, 2.0, 3.0]

    print(f"   Gamma RV: {gamma_rv}")
    print("   CDF values with default parameters:")
    for x in x_values:
        cdf_val = gamma_rv.cdf(x)
        print(f"     CDF({x}) = {cdf_val:.6f}")

    print("   CDF values with overridden parameters (shape=3, scale=2):")
    for x in x_values:
        cdf_val = gamma_rv.cdf(x, shape=3.0, scale=2.0)
        print(f"     CDF({x}) = {cdf_val:.6f}")


if __name__ == "__main__":
    main()
