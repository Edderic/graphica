import logging
from pathlib import Path

import pandas as pd
import pytest

from ..linx.infer import VariableElimination
from ..linx.ds import Query, ConditionalProbabilityTable as CPT, \
    BayesianNetwork
from .conftest import assert_approx_value_df, create_df_easy, create_df_medium, \
    create_df_hard, create_binary_prior_cpt, create_binary_CPT


def test_independence(collider_and_descendant):
    """
    P(X | Y) = P(X)
    """
    bayesian_network = collider_and_descendant

    query = Query(
        outcomes=['X'],
        givens=['Y']
    )

    algo = VariableElimination(
        network=bayesian_network,
        query=query,
    )

    result = algo.compute()

    # independence
    expected_df = pd.DataFrame(
        [
            {'value': 0.7, 'Y': 0, 'X': 0},
            {'value': 0.3, 'Y': 0, 'X': 1},
            {'value': 0.7, 'Y': 1, 'X': 0},
            {'value': 0.3, 'Y': 1, 'X': 1},
        ]
    )

    assert_approx_value_df(
        actual_df=result.df,
        expected_df=expected_df
    )


def test_collider_1(collider_and_descendant):
    """
    P(Z|Y) = ∑ P(Z | x, Y) ⨉ P(x)
             x
    """
    bayesian_network = collider_and_descendant

    query = Query(
        outcomes=['Z'],
        givens=['Y']
    )

    algo = VariableElimination(
        network=bayesian_network,
        query=query
    )

    result = algo.compute()
    expected_df = pd.DataFrame([
        {
            'Z': 0, 'value': 0.55, 'Y': 0
        },
        {
            'Z': 1, 'value': 0.45, 'Y': 0
        },
        {
            'Z': 0, 'value': 0.45, 'Y': 1
        },
        {
            'Z': 1, 'value': 0.55, 'Y': 1
        },
    ])

    assert_approx_value_df(
        actual_df=result.df,
        expected_df=expected_df,
    )
