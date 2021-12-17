import numpy as np
import pandas as pd

from ..linx.infer import VariableElimination
from ..linx.ds import Query, ConditionalProbabilityTable as CPT, \
    BayesianNetwork
from .conftest import assert_approx_value_df, create_df_easy, \
    create_df_medium, create_binary_prior_cpt, create_binary_CPT, \
    create_prior_df


def test_unconnected(two_vars_unconnected_bn):
    """
    P(X | Y) = P(X)
    """
    bayesian_network = two_vars_unconnected_bn

    query = Query(
        outcomes=['X'],
        givens=['Y']
    )

    algo = VariableElimination(
        network=bayesian_network,
        query=query,
    )

    result = algo.compute()

    expected_df = pd.DataFrame([
        {'Y': 0, 'value': 0.1, 'X': 0},
        {'Y': 0, 'value': 0.9, 'X': 1},
        {'Y': 1, 'value': 0.1, 'X': 0},
        {'Y': 1, 'value': 0.9, 'X': 1},
    ])

    result_df = result.get_df()

    assert_approx_value_df(
        actual_df=result_df,
        expected_df=expected_df
    )


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
        actual_df=result.get_df(),
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
        actual_df=result.get_df(),
        expected_df=expected_df,
    )


def test_tree_binary():
    r"""
       A     D
       /\   /\
      v  v v  v
      B   C   E
    """

    cpts = {}

    priors = {
        'A': create_binary_prior_cpt('A', value_for_1=0.7),
        'D': create_binary_prior_cpt('D', value_for_1=0.4),
    }

    easy_cpts = [
        ('A', 'B'),
    ]

    for start_var, end_var in easy_cpts:
        cpts[end_var] = create_binary_CPT(
            vals=[{start_var: 1, 'value': 0.6}, {start_var: 0, 'value': 0.3}],
            outcome=end_var,
            given=start_var
        )

    med_cpts = [
        ('D', 'E'),
    ]

    for start_var, end_var in med_cpts:
        cpts[end_var] = create_binary_CPT(
            vals=[{start_var: 1, 'value': 0.9}, {start_var: 0, 'value': 0.05}],
            outcome=end_var,
            given=start_var
        )

    c_df = pd.DataFrame([
        {"A": 0, "D": 0, "C": 0, "value": 0.2},
        {"A": 0, "D": 0, "C": 1, "value": 0.8},
        {"A": 1, "D": 0, "C": 0, "value": 0.6},
        {"A": 1, "D": 0, "C": 1, "value": 0.8},
        {"A": 1, "D": 1, "C": 0, "value": 0.9},
        {"A": 1, "D": 1, "C": 1, "value": 0.2},
        {"A": 0, "D": 1, "C": 0, "value": 0.1},
        {"A": 0, "D": 1, "C": 1, "value": 0.4},
    ])

    cpts['C'] = CPT(
        df=c_df,
        outcomes=['C'],
        givens=['A', 'D']
    )

    bayesian_network = BayesianNetwork(
        cpts=list(cpts.values()),
        priors=list(priors.values())
    )

    query = Query(
        givens=[
            'E'
        ],
        outcomes=['B']
    )

    variable_elimination = VariableElimination(
        network=bayesian_network,
        query=query
    )

    result = variable_elimination.compute()
    result_df = result.get_df()
    assert_approx_value_df(
        actual_df=result_df[result_df['E'] == 0],
        expected_df=result_df[result_df['E'] == 0],
    )


def test_tree():
    r"""
       A     D
       /\   /\
      v  v v  v
      B   C   E
    """

    cpts = {}

    priors = [
        'A',
        'D'
    ]

    priors = {}
    for prior in priors:
        priors[prior] = CPT(
            df=create_prior_df(outcome=prior),
            outcomes=[prior],
        )

    easy_cpts = [
        ('A', 'B'),
    ]

    for start_var, end_var in easy_cpts:
        cpts[end_var] = CPT(
            df=create_df_easy(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    med_cpts = [
        ('D', 'E'),
    ]

    for start_var, end_var in med_cpts:
        cpts[end_var] = CPT(
            df=create_df_medium(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    c_df = pd.DataFrame([
        {'A': 0, 'D': 0, 'C': 1, 'value': 0.90},
        {'A': 0, 'D': 0, 'C': 0, 'value': 0.10},
        {'A': 0, 'D': 1, 'C': 1, 'value': 0.01},
        {'A': 0, 'D': 1, 'C': 0, 'value': 0.99},
        {'A': 0, 'D': 2, 'C': 1, 'value': 0.02},
        {'A': 0, 'D': 2, 'C': 0, 'value': 0.98},
        {'A': 0, 'D': 3, 'C': 1, 'value': 0.03},
        {'A': 0, 'D': 3, 'C': 0, 'value': 0.97},
        {'A': 0, 'D': 4, 'C': 1, 'value': 0.04},
        {'A': 0, 'D': 4, 'C': 0, 'value': 0.96},
        {'A': 1, 'D': 0, 'C': 1, 'value': 0.80},
        {'A': 1, 'D': 0, 'C': 0, 'value': 0.20},
        {'A': 1, 'D': 1, 'C': 1, 'value': 0.11},
        {'A': 1, 'D': 1, 'C': 0, 'value': 0.89},
        {'A': 1, 'D': 2, 'C': 1, 'value': 0.02},
        {'A': 1, 'D': 2, 'C': 0, 'value': 0.98},
        {'A': 1, 'D': 3, 'C': 1, 'value': 0.03},
        {'A': 1, 'D': 3, 'C': 0, 'value': 0.97},
        {'A': 1, 'D': 4, 'C': 1, 'value': 0.04},
        {'A': 1, 'D': 4, 'C': 0, 'value': 0.96},
        {'A': 2, 'D': 0, 'C': 1, 'value': 0.70},
        {'A': 2, 'D': 0, 'C': 0, 'value': 0.30},
        {'A': 2, 'D': 1, 'C': 1, 'value': 0.11},
        {'A': 2, 'D': 1, 'C': 0, 'value': 0.89},
        {'A': 2, 'D': 2, 'C': 1, 'value': 0.12},
        {'A': 2, 'D': 2, 'C': 0, 'value': 0.88},
        {'A': 2, 'D': 3, 'C': 1, 'value': 0.03},
        {'A': 2, 'D': 3, 'C': 0, 'value': 0.97},
        {'A': 2, 'D': 4, 'C': 1, 'value': 0.04},
        {'A': 2, 'D': 4, 'C': 0, 'value': 0.96},
        {'A': 3, 'D': 0, 'C': 1, 'value': 0.60},
        {'A': 3, 'D': 0, 'C': 0, 'value': 0.40},
        {'A': 3, 'D': 1, 'C': 1, 'value': 0.11},
        {'A': 3, 'D': 1, 'C': 0, 'value': 0.89},
        {'A': 3, 'D': 2, 'C': 1, 'value': 0.12},
        {'A': 3, 'D': 2, 'C': 0, 'value': 0.88},
        {'A': 3, 'D': 3, 'C': 1, 'value': 0.13},
        {'A': 3, 'D': 3, 'C': 0, 'value': 0.87},
        {'A': 3, 'D': 4, 'C': 1, 'value': 0.04},
        {'A': 3, 'D': 4, 'C': 0, 'value': 0.96},
        {'A': 4, 'D': 0, 'C': 1, 'value': 0.50},
        {'A': 4, 'D': 0, 'C': 0, 'value': 0.50},
        {'A': 4, 'D': 1, 'C': 1, 'value': 0.21},
        {'A': 4, 'D': 1, 'C': 0, 'value': 0.79},
        {'A': 4, 'D': 2, 'C': 1, 'value': 0.12},
        {'A': 4, 'D': 2, 'C': 0, 'value': 0.88},
        {'A': 4, 'D': 3, 'C': 1, 'value': 0.13},
        {'A': 4, 'D': 3, 'C': 0, 'value': 0.87},
        {'A': 4, 'D': 4, 'C': 1, 'value': 0.04},
        {'A': 4, 'D': 4, 'C': 0, 'value': 0.96},
    ])
    cpts['C'] = CPT(
        df=c_df,
        outcomes=['C'],
        givens=['A', 'D']
    )

    bayesian_network = BayesianNetwork(
        cpts=list(cpts.values()),
        priors=list(priors.values())
    )

    query = Query(
        givens=[
            'E'
        ],
        outcomes=['B']
    )

    pd.options.display.max_rows = 999

    variable_elimination = VariableElimination(
        network=bayesian_network,
        query=query
    )

    result = variable_elimination.compute()
    result_df = result.get_df()

    for i in range(4):
        # B does not dependent on E
        left = result_df[result_df['E'] == i][['value', 'B']]
        right = result_df[result_df['E'] == i + 1][['value', 'B']]

        assert_approx_value_df(
            actual_df=left,
            expected_df=right
        )


def test_nothing_to_eliminate():
    def create_anenometer_cpt_df(size):
        dfs = []
        minimum = 0
        maximum = 15
        for actual_cfs in np.arange(minimum, maximum, 0.1):
            cubic_feet_per_second = \
                np.random.normal(actual_cfs, 0.1, size=size)

            df = pd.DataFrame({
                'actual_fps': round(actual_cfs, 2),
                'measured_fps': [
                    round(cfs, 1) for cfs in cubic_feet_per_second
                ],
                'value': 0
            })

            dfs.append(df)

        dfs = pd.concat(dfs)

        rates = (
            dfs.groupby(['actual_fps', 'measured_fps']).count()[['value']]
            / dfs.groupby(['actual_fps']).count()[['value']]
        ).reset_index()

        valid_values = rates[(rates['measured_fps'] >= 0)]

        normalized = valid_values\
            .set_index(['actual_fps', 'measured_fps'])[['value']] \
            / valid_values.groupby(['actual_fps']).sum()[['value']]

        return normalized.reset_index()

    def generate_cpts_for_anenometer_readings(
        number_of_readings,
        anenometer_reading_df
    ):
        """
        Generate conditional probability tables of anenometer readings.

        Parameters:
            number_of_readings: integer
            anenometer_reading_df: pd.DataFrame

        Returns: list[CPT]
        """
        cpts = []

        for i in range(number_of_readings):
            cpt = CPT(
                df=anenometer_reading_df.rename(
                    columns={'measured_fps': f'measured_fps_{i}'}
                ),
                outcomes=[f'measured_fps_{i}'],
                givens=['actual_fps']
            )
            cpts.append(cpt)

        return cpts

    def generate_flat_priors(start, end, step, name, round_to=1):
        """
        Parameters:
            start: float
                The lower bound
            end: float
                The upper bound
            step: float
                The increment that will be taken to go from start to end
            round_to: integer
                We round the value so that we can compare floats more easily.
        """
        array = np.arange(start, end, step)

        proba = 1.0 / len(array)

        return pd.DataFrame(
            [{'value': proba, name: round(i, round_to)} for i in array]
        )

    anenometer_reading_df = create_anenometer_cpt_df(10000)

    bn = BayesianNetwork(
        cpts=generate_cpts_for_anenometer_readings(
            number_of_readings=3,
            anenometer_reading_df=anenometer_reading_df
        ),
        priors=[
            CPT(
                df=generate_flat_priors(0, 5, 0.1, 'actual_fps'),
                outcomes=['actual_fps']
            )
        ]
    )

    layer_0_fps = VariableElimination(
        network=bn,
        query=Query(
            outcomes=['actual_fps'],
            givens=[
                {'measured_fps_0': 0.5},
                {'measured_fps_1': 0.7},
                {'measured_fps_2': 0.8}
            ]
        )
    ).compute()

    result_df = layer_0_fps.get_df()
    result_df['value'] / result_df['value'].sum()
    layer_0_fps.normalize(
        [
            'measured_fps_0',
            'measured_fps_1',
            'measured_fps_2'
        ]
    )

    assert result_df[
        result_df['actual_fps'] == 0.7
    ]['value'].values[0] > 0.5
