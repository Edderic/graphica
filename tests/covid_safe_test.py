import graphviz
import pandas as pd
import pytest

from .conftest import assert_approx_value_df
from ..linx.bayesian_network import BayesianNetwork as BN
from ..linx.ds import Query
from ..linx.infer import VariableElimination as VE

from ..linx.examples.covid_safe import (
    create_dose,
    create_dose_small_tmp_df,
    create_dose_tmp_2_df,
    create_dose_tmp_3_df,
    create_dose_tmp_4_df,
    create_dose_tmp_5_df,
    create_dose_df,
    create_num_infected,
    index_name
)


def test_create_dose_small_tmp_df():
    df = create_dose_small_tmp_df(suffix='abc')

    df_2 = create_dose_tmp_2_df(
        suffix='abc',
        unique=df['dose_small_tmp_abc'].unique()
    )

    df_3 = create_dose_tmp_3_df(
        suffix='abc',
        unique=df_2['dose_tmp_2_abc'].unique()
    )

    df_4 = create_dose_tmp_4_df(
        suffix='abc',
        unique=df_3['dose_tmp_3_abc'].unique()
    )

    df_5 = create_dose_tmp_5_df(
        suffix='abc',
        unique=df_4['dose_tmp_4_abc'].unique()
    )

    df_6 = create_dose_tmp_5_df(
        suffix='abc',
        unique=df_5['dose_tmp_5_abc'].unique()
    )


    assert df_6['dose_tmp_5_abc'].describe()['max'] == 5
    assert df_6['dose_tmp_5_abc'].describe()['min'] == 0
    assert df_6['emission_factor_abc'].describe()['min'] >= 1
    assert df_6['emission_factor_abc'].describe()['max'] <= 200


def test_infectious_proba():
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())
    create_num_infected(
        suffix=index_name('edderic', '1', 'work'),
        bayesian_network=bayesian_network
    )
    suffix = index_name('edderic', '1', 'work')
    result = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[f"infectious_proba_{suffix}"],
            givens=[
                {
                    f"reported_frac_given_infectious_{suffix}": 0.1
                },
                {
                    f"ratio_reported_cases_per_pop_{suffix}": 0.00038
                },
                {
                    "infectivity_period_length": 7
                }
            ],
        )
    ).compute()

    result_df = result.get_df()
    assert result_df.shape[0] == 1
    assert result_df.loc[0, 'value'] == 1
    assert result_df.loc[0, f'infectious_proba_{suffix}'] == 0.0266


def test_num_infected():
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())
    create_num_infected(
        suffix=index_name('edderic', '1', 'work'),
        bayesian_network=bayesian_network
    )
    suffix = index_name('edderic', '1', 'work')
    result = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[f"num_infected_{suffix}"],
            givens=[
                {
                    f"reported_frac_given_infectious_{suffix}": 0.1
                },
                {
                    f"ratio_reported_cases_per_pop_{suffix}": 0.00038
                },
                {
                    "infectivity_period_length": 7
                },
                {
                    f"num_people_seen_{suffix}": 2
                },
            ],
        )
    ).compute()


    result_df = result.get_df()
    sort = result_df.sort_values(by=f'num_infected_{suffix}')

    assert sort.loc[0, 'value'] == pytest.approx(0.947508, abs=0.0001)
    assert sort.loc[1, 'value'] == pytest.approx(0.051785, abs=0.0001)
    assert sort.loc[2, 'value'] == pytest.approx(0.000708, abs=0.0001)


def test_num_infected():
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())
    create_num_infected(
        suffix=index_name('edderic', '1', 'work'),
        bayesian_network=bayesian_network
    )
    suffix = index_name('edderic', '1', 'work')
    result = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[f"num_infected_{suffix}"],
            givens=[
                {
                    f"reported_frac_given_infectious_{suffix}": 0.1
                },
                {
                    f"ratio_reported_cases_per_pop_{suffix}": 0.00038
                },
                {
                    "infectivity_period_length": 7
                },
                {
                    f"num_people_seen_{suffix}": 2
                },
            ],
        )
    ).compute()


    result_df = result.get_df()
    sort = result_df.sort_values(by=f'num_infected_{suffix}')

    assert sort.loc[0, 'value'] == pytest.approx(0.947508, abs=0.0001)
    assert sort.loc[1, 'value'] == pytest.approx(0.051785, abs=0.0001)
    assert sort.loc[2, 'value'] == pytest.approx(0.000708, abs=0.0001)


@pytest.mark.f
def test_create_dose():
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())
    suffix = index_name('edderic', '1', 'work')

    create_num_infected(
        suffix=suffix,
        bayesian_network=bayesian_network
    )

    create_dose(
        suffix=suffix,
        bayesian_network=bayesian_network
    )

    result = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[f"dose_{suffix}"],
            givens=[
                {
                    f"reported_frac_given_infectious_{suffix}": 0.1
                },
                {
                    f"ratio_reported_cases_per_pop_{suffix}": 0.00038
                },
                {
                    "infectivity_period_length": 7
                },
                {
                    f"num_people_seen_{suffix}": 2
                },
                {
                    f"mask_inhalation_factor_{suffix}": 1
                },
                {
                    f"mask_exhalation_factor_{suffix}": 1
                },
                {
                    f"volume_{suffix}": 100
                },
                {
                    f"ventilation_factor_{suffix}": 1
                },
                {
                    f"duration_{suffix}": 1
                },
                {
                    f"emission_at_resting_{suffix}": 55
                },
                {
                    f"emission_factor_{suffix}": 1
                },
                {
                    f"inhalation_factor_{suffix}": 1
                },
            ],
        )
    ).compute()

    all_df = result.get_df()

    dose_and_proba_df = all_df[
        [f'dose_{suffix}', 'value']
    ].sort_values(by=f'dose_{suffix}')

    expected_df = pd.DataFrame(
        [
            {
                f'dose_{suffix}': 0,
                'value': 0.947508
            },
            {
                f'dose_{suffix}': 0.1584,
                'value': 0.051785
            },
            {
                f'dose_{suffix}': 0.3168,
                'value': 0.000708
            }
        ]
    )

    assert_approx_value_df(
        dose_and_proba_df,
        expected_df
    )
