import numpy as np
from ..linx.ds import BayesianNetwork as BN
from ..linx.random.beta import Beta
from ..linx.random.binomial import Binomial
from ..linx.inference.metropolis_hastings import MetropolisHastings
from ..linx.query import Query


def test_metropolis_hastings_beta_binomial_posterior():
    """
    Test MetropolisHastings with Beta(1,1) prior and Binomial(200, p) likelihood
    with 100 successes and 100 failures. The posterior should be Beta(101, 101),
    so the 95% credible interval should be close to (0.45, 0.55).
    """
    np.random.seed(42)

    # Set up the Bayesian network
    bn = BN()

    # Beta prior for p
    prior = Beta(name='p', alpha=1, beta_param=1)
    bn.add_node(prior)

    # Binomial likelihood (fixed observed data)
    n = 200
    k = 100
    likelihood = Binomial(name='x', n=n, p=prior)  # p will be replaced in sampling
    bn.add_node(likelihood)

    # Define a transition function for Metropolis-Hastings
    def transition(particle):
        # Propose new p from a normal centered at current p
        current_p = particle.get_value('p')
        proposal_p = np.clip(np.random.normal(current_p, 0.05), 0, 1)
        new_particle = particle.copy()
        new_particle.set_value('p', proposal_p)
        # x is fixed (observed)
        new_particle.set_value('x', k)
        return new_particle

    # Query: condition on x=100
    query = Query(outcomes=['p'], givens=[{'x': k}])

    # Initial particle
    initial_particle = None

    # Metropolis-Hastings sampler
    sampler = MetropolisHastings(
        network=bn,
        query=query,
        transition_function=transition,
        initial_particle=initial_particle
    )

    # Run sampler
    samples = sampler.sample(n=2000, burn_in=500)
    p_samples = np.array([particle.get_value('p') for particle in samples])

    # Compute 95% credible interval
    lower, upper = np.percentile(p_samples, [2.5, 97.5])
    mean = np.mean(p_samples)

    # Theoretical posterior is Beta(101, 101)
    from scipy.stats import beta
    beta_dist = beta(101, 101)
    expected_lower, expected_upper = beta_dist.ppf([0.025, 0.975])
    expected_mean = beta_dist.mean()

    print(f"Sampled mean: {mean:.4f}, 95% CI: ({lower:.4f}, {upper:.4f})")
    print(f"Expected mean: {expected_mean:.4f}, 95% CI: ({expected_lower:.4f}, {expected_upper:.4f})")

    # Assert that the credible interval is close to the theoretical one
    assert abs(mean - expected_mean) < 0.02
    assert abs(lower - expected_lower) < 0.03
    assert abs(upper - expected_upper) < 0.03
