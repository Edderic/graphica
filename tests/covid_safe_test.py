import graphviz
import pandas as pd
import pytest

from .conftest import assert_approx_value_df
from ..linx.bayesian_network import BayesianNetwork as BN
from ..linx.ds import Query
from ..linx.infer import VariableElimination as VE
from ..linx.data import InMemoryData

from ..linx.examples.covid_safe import (
    create_dose_from_strangers,
    create_dose,
    create_dose_small_tmp_df,
    create_dose_tmp_2_df,
    create_dose_tmp_3_df,
    create_dose_tmp_4_df,
    create_dose_tmp_5_df,
    create_dose_df,
    create_num_infected,
    divide_1_by,
    index_name
)

from ..linx.misc import get_tmp_path, clean_tmp


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


def test_create_volume_lambda_df():
    new_suffix = index_name('1', 'friends', 'person 1', 'person 2')
    suffix_1 = index_name('1', 'friends')

    df = create_volume_lambda_df(
        suffix=suffix_1,
        new_key=f'tmp_{new_suffix}'
    )

    assert df[f'lambda_{suffix_1}'].shape[0] > 0
    assert df[f'volume_{suffix_1}'].shape[0] > 0
    assert df[f'tmp_{new_suffix}'].shape[0] > 0
    # multiplication happened
    assert df[df[f'tmp_{new_suffix}'] == 2880 * 480].shape[0] == 1
    assert all(df['value']) == 1


def test_divide_1_by():
    suffix_1 = index_name('1', 'friends')

    df = create_volume_lambda_df(
        suffix=suffix_1,
        new_key=f'tmp_{suffix_1}'
    )

    division_df = divide_1_by(
        divisor_unique=df[f'tmp_{suffix_1}'].unique(),
        divisor_name=f'tmp_{suffix_1}',
        new_key=f'tmp_2_{suffix_1}'
    )

    assert all(division_df['value']) == 1
    assert division_df[f'tmp_{suffix_1}'].shape[0] > 0
    assert division_df[f'tmp_2_{suffix_1}'].shape[0] > 0

    assert division_df[
        (division_df[f'tmp_{suffix_1}'] == 1)
        & (division_df[f'tmp_2_{suffix_1}'] == 1)
    ].shape[0] == 1


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


def test_create_at_least_one_inf():
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())

    tmp = get_tmp_path()

    storage_folder = tmp / "somewhere"
    storage_folder.mkdir(exist_ok=True)
    clean_tmp(storage_folder)

    create_dose_from_strangers(
        time=1,
        person='edderic',
        event='work',
        bayesian_network=bayesian_network,
        storage_folder=storage_folder
    )

    suffix_1 = index_name(1, 'edderic', 'work')
    date_event_others_to_person = index_name(1, 'work', 'others', 'edderic')
    date_event_self_suffix = index_name(1, 'work', 'edderic')
    date_self_suffix = index_name(1, 'edderic')
    day = 1
    day_index = index_name(day)
    outcome_col = f"dose_tmp_13_{date_event_others_to_person}"

    result_1 = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f"volume_{suffix_1}": 20
                },
                {
                    f"ventilation_{suffix_1}": 4
                },
                {
                    f"activity_exhalation_{date_event_others_to_person}": "Resting - Speaking"
                },
                {
                    "age_edderic": '31 to <41'
                },
                {
                    f"activity_{date_event_self_suffix}": 'Sedentary/Passive'
                },
                {
                    f"duration_{date_event_others_to_person}": 5
                },
                {
                    f"quanta_{date_self_suffix}": 50
                },
                {
                    f"mask_quality_{date_event_others_to_person}": 0
                },
                {
                    f"perc_masked_{date_event_others_to_person}": 0
                },
                {
                    f"num_positive_cases_{day_index}": 2000.0
                },
                {
                    f"pop_size_{day_index}": 60_000_000
                },
                {
                    f"num_days_inf_{day_index}": 10
                },
                {
                    f"unreported_positive_{day_index}": 10
                },
                {
                    f"num_people_seen_{date_event_others_to_person}": 4
                },
            ],
        )
    ).compute()

    # Positivity rate 5 times higher than the first scenario.
    result_2 = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f"volume_{suffix_1}": 20
                },
                {
                    f"ventilation_{suffix_1}": 4
                },
                {
                    f"activity_exhalation_{date_event_others_to_person}": "Resting - Speaking"
                },
                {
                    "age_edderic": '31 to <41'
                },
                {
                    f"activity_{date_event_self_suffix}": 'Sedentary/Passive'
                },
                {
                    f"duration_{date_event_others_to_person}": 5
                },
                {
                    f"quanta_{date_self_suffix}": 50
                },
                {
                    f"mask_quality_{date_event_others_to_person}": 0
                },
                {
                    f"perc_masked_{date_event_others_to_person}": 0
                },
                {
                    f"num_positive_cases_{day_index}": 10000.0
                },
                {
                    f"pop_size_{day_index}": 60_000_000
                },
                {
                    f"num_days_inf_{day_index}": 10
                },
                {
                    f"unreported_positive_{day_index}": 10
                },
                {
                    f"num_people_seen_{date_event_others_to_person}": 4
                },
            ],
        )

    ).compute()

    # See 30 people instead of 4 in the first scenario
    result_3 = VE(
        network=bayesian_network,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f"volume_{suffix_1}": 60
                },
                {
                    f"ventilation_{suffix_1}": 4
                },
                {
                    f"activity_exhalation_{date_event_others_to_person}": "Resting - Speaking"
                },
                {
                    "age_edderic": '31 to <41'
                },
                {
                    f"activity_{date_event_self_suffix}": 'Sedentary/Passive'
                },
                {
                    f"duration_{date_event_others_to_person}": 5
                },
                {
                    f"quanta_{date_self_suffix}": 50
                },
                {
                    f"mask_quality_{date_event_others_to_person}": 0
                },
                {
                    f"perc_masked_{date_event_others_to_person}": 0
                },
                {
                    f"num_positive_cases_{day_index}": 2000.0
                },
                {
                    f"pop_size_{day_index}": 60_000_000
                },
                {
                    f"num_days_inf_{day_index}": 10
                },
                {
                    f"unreported_positive_{day_index}": 10
                },
                {
                    f"num_people_seen_{date_event_others_to_person}": 30
                },
            ],
        )
    ).compute()
    result_1_df = result_1.get_df()
    result_2_df = result_2.get_df()
    result_3_df = result_3.get_df()
    df_1 = result_1_df[[outcome_col, 'value']]
    df_2 = result_2_df[[outcome_col, 'value']]
    df_3 = result_3_df[[outcome_col, 'value']]

    clean_tmp(storage_folder)

    assert df_1[df_1[outcome_col] == 0].loc[0, 'value'] \
        > df_2[df_2[outcome_col] == 0].loc[0, 'value']

    assert df_1[df_1[outcome_col] == 0].loc[0, 'value'] \
        > df_3[df_2[outcome_col] == 0].loc[0, 'value']


@pytest.mark.f
def test_create_at_least_one_inf_1():
    """
    When people are masked, then they are more likely to be safe.
    """
    bayesian_network_1 = BN(graphviz_dag=graphviz.Digraph())
    bayesian_network_2 = BN()

    date_person_event_suffix = index_name(1, 'edderic', 'work')
    date_event_others_to_person = index_name(1, 'work', 'others', 'edderic')
    date_event_self_suffix = index_name(1, 'work', 'edderic')
    self_suffix = index_name('edderic')
    day = 1
    day_index = index_name(day)
    outcome_col = f"dose_tmp_13_{date_event_others_to_person}"

    results = []

    for bn in [bayesian_network_1, bayesian_network_2]:
        create_dose_from_strangers(
            time=1,
            person='edderic',
            event='work',
            bayesian_network=bn,
            storage_folder=None
        )

        dictionary = {
            f"volume_{date_person_event_suffix}": {
                60: 1.0
            },
            f"ventilation_{date_person_event_suffix}": {
                4: 1.0
            },
            f"activity_exhalation_{date_event_others_to_person}": {
                "Resting - Speaking": 1.0
            },
            f"age_{self_suffix}": {
                '31 to <41': 1.0
            },
            f"activity_{date_event_self_suffix}": {
                'Sedentary/Passive': 1.0
            },
            f"duration_{date_event_others_to_person}": {
                5: 1.0
            },
            f"quanta_{date_event_others_to_person}": {
                50: 1.0
            },
            f"mask_quality_{date_event_others_to_person}": {
                0: 1.0
            },
            f"mask_{date_event_self_suffix}": {
                1.0: 1.0
            },
            f"perc_masked_{date_event_others_to_person}": {
                0: 1.0
            },
            f"num_positive_cases_{day_index}": {
                2000.0: 1.0
            },
            f"pop_size_{day_index}": {
                60_000_000: 1.0
            },
            f"num_days_inf_{day_index}": {
                10: 1.0
            },
            f"unreported_positive_{day_index}": {
                10: 1.0
            },
            f"num_people_seen_{date_event_others_to_person}": {
                30: 1.0
            },
        }

        if bn == bayesian_network_2:
            dictionary[
                f"mask_{date_event_self_suffix}"
            ] = {
                0.01: 1.0
            }

        bn.set_priors(
            dictionary=dictionary,
            data_class=InMemoryData,
        )

        result = VE(
            network=bn,
            query=Query(
                outcomes=[outcome_col],
                givens=[],
            )
        ).compute()

        results.append(result.get_df())

    assert results[0].loc[0, outcome_col] == results[1].loc[0, outcome_col]
    assert (results[0][outcome_col] * results[0]['value']).sum() > \
        (results[1][outcome_col] * results[1]['value']).sum()
