import pandas as pd

from ..linx.infer import VariableElimination
from .conftest import assert_approx_value_df


def test_independence(collider_and_descendant):
    bayesian_network = collider_and_descendant

    algo = VariableElimination(
        network=bayesian_network,
        outcomes=['X'],
        given=['Y']
    )

    result = algo.compute()

    # independence
    expected_df = pd.DataFrame(
        [
            {'count': 0.7, 'Y': 0, 'X': 0},
            {'count': 0.3, 'Y': 0, 'X': 1},
            {'count': 0.7, 'Y': 1, 'X': 0},
            {'count': 0.3, 'Y': 1, 'X': 1},
        ]
    )

    assert_approx_value_df(
        actual_df=result.df,
        expected_df=expected_df
    )


def test_collider(collider_and_descendant):
    """
    P(Z|Y) = ∑ P(Z | x, Y) ⨉ P(x)
             x
    """
    bayesian_network = collider_and_descendant

    algo = VariableElimination(
        network=bayesian_network,
        outcomes=['Z'],
        given=['Y']
    )

    result = algo.compute()
    expected_df = pd.DataFrame([
        {
            'Z': 0, 'count': 0.55, 'Y': 0
        },
        {
            'Z': 1, 'count': 0.45, 'Y': 0
        },
        {
            'Z': 0, 'count': 0.45, 'Y': 1
        },
        {
            'Z': 1, 'count': 0.55, 'Y': 1
        },
    ])

    assert_approx_value_df(
        actual_df=result.df,
        expected_df=expected_df,
    )
