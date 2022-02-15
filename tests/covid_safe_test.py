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
    create_infection_from_dose,
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

    assert results[0].loc[0, outcome_col] == \
        results[1].loc[0, outcome_col]

    assert (results[0][outcome_col] * results[0]['value']).sum() > \
        (results[1][outcome_col] * results[1]['value']).sum()


def test_create_infection_from_dose():
    """
    Test that each possible dose affects the probability of infection.
    """
    suffix = index_name('day_1', 'person_1')
    df = create_infection_from_dose(suffix)

    assert df[
        (df[f'dose_{suffix}'] == 0) &
        (df[f'infected_{suffix}'] == 0)
    ].iloc[0]['value'] == 1

    assert df[
        (df[f'dose_{suffix}'] == 0) &
        (df[f'infected_{suffix}'] == 1)
    ].iloc[0]['value'] == 0

    # high end
    assert df[
        (df[f'dose_{suffix}'] == 9.9999) &
        (df[f'infected_{suffix}'] == 0)
    ].iloc[0]['value'].round(6) == 0.000045
    assert df[
        (df[f'dose_{suffix}'] == 9.9999) &
        (df[f'infected_{suffix}'] == 1)
    ].iloc[0]['value'].round(6) == 0.999955
