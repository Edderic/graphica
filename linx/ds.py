"""
Data Structures module.

Classes:
    - ConditionalProbabilityTable
    - DirectedAcyclicGraph
    - LogFactorAdapter
    - Factor
    - Factors
    - BayesianNetwork
    - MarkovNetwork
"""
from .query import Query
from .conditional_probability_table import ConditionalProbabilityTable
from .directed_acyclic_graph import DirectedAcyclicGraph
from .factor import Factor
from .log_factor_adapter import LogFactorAdapter
from .factors import Factors
from .markov_network import MarkovNetwork
from .bayesian_network import BayesianNetwork
