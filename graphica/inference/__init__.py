"""
Inference module for Bayesian network sampling.
"""

from .metropolis_hastings import MetropolisHastings
from .default_transition import DefaultTransition

__all__ = ["MetropolisHastings", "DefaultTransition"]
