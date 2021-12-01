import pandas as pd
import pytest

from ..linx.infer import VariableElimination


def test_elimination(collider_and_descendant):
    bayesian_network = collider_and_descendant

    algo = VariableElimination(
        network=bayesian_network,
        outcomes=['X'],
        given=['Y']
    )

    result = algo.compute()
    columns = ['X', 'Y', 'count']
    expected_df = pd.DataFrame(
        [
            {'count': 0.7, 'Y': 0, 'X': 0},
            {'count': 0.3, 'Y': 0, 'X': 1},
            {'count': 0.7, 'Y': 1, 'X': 0},
            {'count': 0.3, 'Y': 1, 'X': 1},
        ]
    )[columns].sort_values(by=columns)

    sorted_result = result\
        .df\
        .sort_values(
            by=columns
        )[columns]

    for (_, x), (_, y) in zip(
        sorted_result.iterrows(),
        expected_df.iterrows()
    ):
        assert x['count'] == pytest.approx(
            y['count'],
            abs=0.01
        )

        assert x['X'] == pytest.approx(
            y['X'],
            abs=0.01
        )

        assert x['Y'] == pytest.approx(
            y['Y'],
            abs=0.01
        )
