import graphviz
import pytest

from ..linx.bayesian_network import BayesianNetwork as BN
from ..linx.ds import Query
from ..linx.infer import VariableElimination as VE

from ..linx.examples.covid_safe import (
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
#
#
# def test_create_dose_covid():
    # df = create_dose_covid_df(suffix='abc')
    # df.memory_usage()
    # import pdb; pdb.set_trace()
#
