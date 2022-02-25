import graphviz
import pandas as pd
import pytest

from .conftest import assert_approx_value_df
from ..linx.bayesian_network import BayesianNetwork as BN
from ..linx.ds import Query
from ..linx.infer import VariableElimination as VE
from ..linx.data import InMemoryData

from ..linx.examples.covid_safe import (
    create_days_since_infection_covid,
    create_dose_from_strangers,
    create_immunity_factor,
    create_infection_from_dose,
    create_inf_dsi_viral_load_measurements,
    create_longitudinal,
    create_viral_load,
    create_viral_load_n,
    create_viral_load_p,
    create_volume_ventilation_df,
    divide_1_by,
    index_name
)

from ..linx.misc import get_tmp_path, clean_tmp


def test_divide_1_by():
    suffix_1 = index_name('1', 'friends')

    df = create_volume_ventilation_df(
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


def test_create_at_least_one_inf_1():
    """
    When people are masked, then they are more likely to be safe.
    """
    bayesian_network_1 = BN(graphviz_dag=graphviz.Digraph())
    bayesian_network_2 = BN()

    time_format = '%m-%d-%y'
    date = pd.date_range(start='02-23-2022', end='02-23-2022')[0]
    date_str = date.strftime(time_format)
    date_person_event_suffix = index_name(date_str, 'edderic', 'work')
    date_event_others_to_person = index_name(
        date_str,
        'work',
        'others',
        'edderic'
    )
    date_event_self_suffix = index_name(date_str, 'work', 'edderic')
    self_suffix = index_name('edderic')
    day_index = index_name(date_str)
    outcome_col = f"dose_tmp_13_{date_event_others_to_person}"

    results = []

    for bn in [bayesian_network_1, bayesian_network_2]:
        create_dose_from_strangers(
            time=date,
            time_format=time_format,
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

    assert results[0].loc[0, outcome_col] == \
        results[1].loc[0, outcome_col]

    assert (results[0][outcome_col] * results[0]['value']).sum() > \
        (results[1][outcome_col] * results[1]['value']).sum()


@pytest.mark.f
def test_create_infection_from_dose():
    """
    Test that each possible dose affects the probability of infection.
    """
    suffix = index_name('day_1', 'person_1')
    dose_key = f'dose_{suffix}'
    infected_key = f'infected_{suffix}'

    df = create_infection_from_dose(
        suffix,
        dose_key=dose_key,
        infected_key=infected_key,
    )

    assert df[
        (df[dose_key] == 0) &
        (df[infected_key] == 0)
    ].iloc[0]['value'] == 1

    assert df[
        (df[dose_key] == 0) &
        (df[infected_key] == 1)
    ].iloc[0]['value'] == 0

    # high end
    assert df[
        (df[dose_key] == 9.9999) &
        (df[infected_key] == 0)
    ].iloc[0]['value'].round(6) == 0.000045
    assert df[
        (df[dose_key] == 9.9999) &
        (df[infected_key] == 1)
    ].iloc[0]['value'].round(6) == 0.999955


def test_create_days_since_infection_covid():
    pre_suffix = index_name(1, 'edderic')
    suffix = index_name(2, 'edderic')

    pre_dsi_key = f'dsi_{pre_suffix}'
    dsi_key = f'dsi_{suffix}'
    infected_key = f'infected_{suffix}'
    max_num_days_since_infection = 21

    df = create_days_since_infection_covid(
        dsi_key,
        pre_dsi_key,
        infected_key,
        max_num_days_since_infection=max_num_days_since_infection
    )

    # If person wasn't infected the day before, and the person didn't get
    # infected for the day, then the person should have 0 for the dsi
    assert df[
        (df[pre_dsi_key] == 0) &
        (df[infected_key] == 0)
    ].iloc[0][dsi_key] == 0

    # If person wasn't infected the day before, and the person DID get
    # infected for the day, then the person should have 1 for the dsi
    assert df[
        (df[pre_dsi_key] == 0) &
        (df[infected_key] == 1)
    ].iloc[0][dsi_key] == 1

    # If the previous day was the 27th day, then we reset back to 0
    # (susceptible)
    assert df[
        (df[pre_dsi_key] == max_num_days_since_infection - 1)
    ].iloc[0][dsi_key] == 0

    assert df[
        (df[pre_dsi_key] == max_num_days_since_infection - 1)
    ].iloc[1][dsi_key] == 0


def test_create_viral_load_n():
    suffix = index_name(1, 'edderic')
    viral_load_n_key = f'viral_load_n_{suffix}'
    immunity_key = f'immunity_{suffix}'

    df = create_viral_load_n(
        viral_load_n_key=viral_load_n_key,
        immunity_key=immunity_key
    )

    assert df[
        (df[viral_load_n_key] == 10) &
        (df[immunity_key] == 1)
    ].iloc[0]['value'] == 0.2


def test_create_viral_load():
    suffix = index_name(1, 'edderic')
    person_index = index_name('edderic')
    dsi_key = f'dsi_{suffix}'
    viral_load_n_key = f'viral_load_n_{suffix}'
    viral_load_p_key = f'viral_load_p_{suffix}'
    immunity_factor_key = f'immunity_factor_{person_index}'
    viral_load_key = f'viral_load_{suffix}'

    df = create_viral_load(
        unique_n=[10, 11, 12, 13, 14, 15, 16],
        unique_p=[0.6],
        unique_immunity_factor=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        dsi_key=dsi_key,
        immunity_factor_key=immunity_factor_key,
        viral_load_key=viral_load_key,
        n_key=viral_load_n_key,
        p_key=viral_load_p_key,
        max_num_days_since_infection=21
    )

    assert all(
        df[
            (df[f'dsi_{suffix}'] == 0)
        ][f'viral_load_{suffix}'] == 0
    )


def test_create_inf_dsi_viral_load_measurements_1():
    """
    Test that getting a positive PCR makes it more likely that someone has a
    high viral load than when we don't know the PCR result.
    """
    bayesian_network_1 = BN(graphviz_dag=graphviz.Digraph())

    person = 'edderic'
    date = pd.date_range(start='02/23/22', end='02/23/22')[0]
    time_format = '%m-%d-%y'
    date_str = date.strftime(time_format)
    person_index = index_name(person)
    date_event_others_to_person = index_name(
        date_str,
        'work',
        'others',
        person
    )
    date_self_suffix = index_name(date_str, person)
    dose_key = f"dose_tmp_13_{date_event_others_to_person}"
    outcome_col = f'viral_load_{date_self_suffix}'

    bn = bayesian_network_1

    viral_load_n_key = f'viral_load_n_{person_index}'
    viral_load_p_key = f'viral_load_p_{person_index}'

    immunity_key = f'immunity_{person_index}'

    viral_load_n_df = create_viral_load_n(
        viral_load_n_key=viral_load_n_key,
        immunity_key=immunity_key
    )

    viral_load_p_df = create_viral_load_p(
        viral_load_p_key=viral_load_p_key,
        immunity_key=immunity_key
    )
    immunity_factor_key = f'immunity_factor_{person_index}'
    immunity_factor_df = create_immunity_factor(
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key
    )

    create_inf_dsi_viral_load_measurements(
        person=person,
        time=date,
        dose_key=dose_key,
        bayesian_network=bayesian_network_1,
        time_format=time_format,
        viral_load_n_key=viral_load_n_key,
        viral_load_p_key=viral_load_p_key,
        viral_load_n_df=viral_load_n_df,
        viral_load_p_df=viral_load_p_df,
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key,
        immunity_factor_df=immunity_factor_df,
    )

    result = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'pcr_{date_self_suffix}': 0
                }
            ],
        )
    ).compute()

    result_df = result.get_df()
    mean_1 = (result_df[outcome_col] * result_df['value']).sum()

    result_2 = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'pcr_{date_self_suffix}': 1
                }
            ],
        )
    ).compute()

    result_df_2 = result_2.get_df()
    mean_2 = (result_df_2[outcome_col] * result_df_2['value']).sum()

    assert mean_1 < mean_2


def test_create_inf_dsi_viral_load_measurements_2():
    """
    Test that getting a positive rapid test should make us more certain that
    someone has a high viral load than getting a positive PCR.
    """
    bayesian_network_1 = BN(graphviz_dag=graphviz.Digraph())

    person = 'edderic'
    person_index = index_name(person)
    date = pd.date_range(start='02/23/22', end='02/23/22')[0]
    time_format = '%m-%d-%y'
    date_str = date.strftime(time_format)
    date_event_others_to_person = index_name(
        date_str,
        'work',
        'others',
        person
    )
    date_self_suffix = index_name(date_str, person)

    dose_key = f"dose_tmp_13_{date_event_others_to_person}"
    outcome_col = f'viral_load_{date_self_suffix}'

    bn = bayesian_network_1
    viral_load_n_key = f'viral_load_n_{person_index}'
    viral_load_p_key = f'viral_load_p_{person_index}'

    immunity_key = f'immunity_{person_index}'

    viral_load_n_df = create_viral_load_n(
        viral_load_n_key=viral_load_n_key,
        immunity_key=immunity_key
    )

    viral_load_p_df = create_viral_load_p(
        viral_load_p_key=viral_load_p_key,
        immunity_key=immunity_key
    )
    immunity_factor_key = f'immunity_factor_{person_index}'
    immunity_factor_df = create_immunity_factor(
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key
    )

    create_inf_dsi_viral_load_measurements(
        person=person,
        time=date,
        dose_key=dose_key,
        bayesian_network=bayesian_network_1,
        time_format=time_format,
        viral_load_n_key=viral_load_n_key,
        viral_load_p_key=viral_load_p_key,
        viral_load_n_df=viral_load_n_df,
        viral_load_p_df=viral_load_p_df,
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key,
        immunity_factor_df=immunity_factor_df,
    )

    result = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'rapid_{date_self_suffix}': 1
                }
            ],
        )
    ).compute()

    result_df = result.get_df()
    rapid_mean = (result_df[outcome_col] * result_df['value']).sum()

    result_2 = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'pcr_{date_self_suffix}': 1
                }
            ],
        )
    ).compute()

    result_df_2 = result_2.get_df()
    pcr_mean = (result_df_2[outcome_col] * result_df_2['value']).sum()

    assert rapid_mean > pcr_mean


def test_create_inf_dsi_viral_load_measurements_4():
    """
    Rapid tests show positive when viral loads are high. In contrast, PCR is
    more sensitive, and could give a positive result, even when person is not
    infectious. So if we have two people, and one of them tests positive in a
    PCR, while the other tests positive in a rapid test, the latter's mean day
    since infection should be lower than the former's day since infection.

    """
    bayesian_network_1 = BN(graphviz_dag=graphviz.Digraph())

    person = 'edderic'
    person_index = index_name(person)
    time_format = '%m-%d-%y'
    date = pd.date_range(start='02/23/2022', end='02/23/2022')[0]
    date_str = date.strftime(time_format)
    date_event_others_to_person = index_name(
        date_str,
        'work',
        'others',
        person
    )
    date_self_suffix = index_name(date_str, person)
    dose_key = f"dose_tmp_13_{date_event_others_to_person}"
    outcome_col = f'dsi_{date_self_suffix}'

    bn = bayesian_network_1

    viral_load_n_key = f'viral_load_n_{person_index}'
    viral_load_p_key = f'viral_load_p_{person_index}'

    immunity_key = f'immunity_{person_index}'

    viral_load_n_df = create_viral_load_n(
        viral_load_n_key=viral_load_n_key,
        immunity_key=immunity_key
    )

    viral_load_p_df = create_viral_load_p(
        viral_load_p_key=viral_load_p_key,
        immunity_key=immunity_key
    )
    immunity_factor_key = f'immunity_factor_{person_index}'
    immunity_factor_df = create_immunity_factor(
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key
    )

    create_inf_dsi_viral_load_measurements(
        person=person,
        time=date,
        dose_key=dose_key,
        bayesian_network=bayesian_network_1,
        time_format=time_format,
        viral_load_n_key=viral_load_n_key,
        viral_load_p_key=viral_load_p_key,
        viral_load_n_df=viral_load_n_df,
        viral_load_p_df=viral_load_p_df,
        immunity_key=immunity_key,
        immunity_factor_key=immunity_factor_key,
        immunity_factor_df=immunity_factor_df,
    )

    result = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'rapid_{date_self_suffix}': 1
                }
            ],
        )
    ).compute()

    result_df = result.get_df()
    rapid_mean = (result_df[outcome_col] * result_df['value']).sum()

    result_2 = VE(
        network=bn,
        query=Query(
            outcomes=[outcome_col],
            givens=[
                {
                    f'pcr_{date_self_suffix}': 1
                }
            ],
        )
    ).compute()

    result_df_2 = result_2.get_df()
    pcr_mean = (result_df_2[outcome_col] * result_df_2['value']).sum()

    assert rapid_mean < pcr_mean


