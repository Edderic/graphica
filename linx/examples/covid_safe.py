"""
CovidSafe: A Risk Estimation to make social gatherings safer.
"""
import graphviz
import numpy as np
import pandas as pd
from scipy.stats import binom


from ..ds import BayesianNetwork as BN, \
    ConditionalProbabilityTable as CPT
from ..query import Query
from ..data import InMemoryData, ParquetData
from ..infer import VariableElimination as VE


def name(title, i, j=None, k=None):
    """
    Provides formatting for a name, given title and up to 3 indices.

    Parameters:
        title: string
        i: string
        j: string
        k: string

    Returns: string
    """
    if i is not None and j is None and k is None:
        return f"{title}_({i})"
    if i is not None and j is not None and k is None:
        return f"{title}_({i}, {j})"
    if i is not None and j is not None and k is not None:
        return f"{title}_({i}, {j}, {k})"


def index_name(i, j=None, k=None, l=None):
    """
    Provides formatting for a name, given title and up to 3 indices.

    Parameters:
        title: string
        i: string
        j: string
        k: string
        l: string

    Returns: string
    """
    if i is not None and j is None and k is None and l is None:
        return f"({i})"
    if i is not None and j is not None and k is None and l is None:
        return f"({i}, {j})"
    if i is not None and j is not None and k is not None and l is None:
        return f"({i}, {j}, {k})"

    if i is not None and j is not None and k is not None and l is not None:
        return f"({i}, {j}, {k}, {l})"


def add_edge_to_bn(bn, df, outcome_var, storage_folder):
    """
    Add dataframe to bayesian network by first wrapping it with a CPT.
    """
    givens = list(set(df.columns) - {'value', outcome_var})
    bn.add_edge(
        CPT(
            InMemoryData(
                df,
                storage_folder
            ),
            givens=givens,
            outcomes=[outcome_var]
        )
    )


def create_vaccination_prior(
    bayesian_network,
    suffix,
    vaccination_rate=0.75
):
    """
    Add a vaccination prior that can be made more specific via the suffix
    parameter.

    Parameters:
        bayesian_network: linx.bayesian_network.BayesianNetwork

        suffix: string
            A string for personalization.

        vaccination_rate: float
            Between 0 and 1.
    """
    outcome_name = f'vax_{suffix}'
    df = pd.DataFrame([
        {
            outcome_name: 1,
            'value': 0.75
        },
        {
            outcome_name: 0,
            'value': 0.25
        }
    ])

    add_edge_to_bn(
        bayesian_network,
        df=df,
        outcome_var=outcome_name,
        storage_folder=None
    )


def create_days_since_infection_covid(
    bayesian_network,
    suffix,
    dose_df,
    person,
    day,
):
    """
    Create days of infection for COVID.

    If someone is susceptible and gets a strong enough dose, then the dose
    determines their day-of-infection status.

    If someone is already infected recently, then we just add 1 to it.

    Parameters:
        bayesian_network: linx.bayesian_network.BayesianNetwork

        suffix: string
            A string for personalization.

    """
    collection = []

    num_days_of_infection = 30
    days_of_infection = list(range(num_days_of_infection)) + ["Susceptible"]
    previous_day_of_infection = list(days_of_infection)
    # dsi: days since infection event
    outcome_name = f'dsi_{name(person, day)}'
    prev_outcome_name = f'dsi_{name(person, day - 1)}'

    for p in previous_day_of_infection:
        if isinstance(p, int):
            if p != num_days_of_infection - 1:
                previous_day_of_infection = str(p)
                current_day_of_infection = str(p + 1)
            else:
                previous_day_of_infection = str(p)
                current_day_of_infection = "Susceptible"

            collection.append(
                {
                    prev_outcome_name: previous_day_of_infection,
                    outcome_name: current_day_of_infection,
                    'value': 1
                }
            )

    df = pd.DataFrame(collection)

    copy = dose_proba.copy()
    copy.loc[:, 'value'] = (1 - np.exp(-dose_df[f'dose_{suffix}'] * dose_df['value']))
    copy[prev_outcome_name] = "Susceptible"
    copy[outcome_name] = 0

    concat_df = pd.concat(
        [
            df,
            copy
        ]
    )

    add_edge_to_bn(
        bayesian_network,
        df=concat_df,
        outcome_var=outcome_name,
        storage_folder=None
    )


def mega_join_cross_product(parameters, dtypes):
    keys = list(parameters.keys())

    collection = pd.DataFrame([])

    for i in range(len(keys) - 1):
        key_1 = keys[i]
        key_2 = keys[i + 1]

        if collection.shape[0] == 0:
            df_1 = pd.DataFrame({
                key_1: parameters[key_1],
                'value': 1
            })

            df_1[key_1] = df_1[key_1].astype(dtypes[key_1])

            df_1['tmp'] = 1
            df_1['tmp'] = df_1['tmp'].astype('bool')

            collection = df_1

        df_2 = pd.DataFrame({
            key_2: parameters[key_2]
        })

        df_2['tmp'] = 1
        df_2['tmp'] = df_2['tmp'].astype('bool')

        df_2[key_2] = df_2[key_2].astype(dtypes[key_2])

        collection = collection.merge(df_2, on='tmp')

    return collection.drop(columns=['tmp'])


def create_dose_small_tmp_df(suffix):
    parameters = {
        f'mask_exhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
        f'mask_inhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
        f'volume_{suffix}':
            [20, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880],
        f'ventilation_factor_{suffix}': [0.1, 0.5, 1, 2, 3, 5, 8, 13, 21, 34],
        f'inhalation_at_rest_{suffix}': [0.288]
    }

    dtypes = {
        f'mask_exhalation_factor_{suffix}': 'float64',
        f'mask_inhalation_factor_{suffix}': 'float64',
        f'volume_{suffix}':
            'int16',
        f'ventilation_factor_{suffix}': 'float64',
        f'inhalation_at_rest_{suffix}': 'float64'
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_small_tmp_{suffix}'] = \
        collection[f'mask_exhalation_factor_{suffix}'] * \
        collection[f'mask_inhalation_factor_{suffix}'] * \
        collection[f'inhalation_at_rest_{suffix}'] / \
        collection[f'volume_{suffix}'] / \
        collection[f'ventilation_factor_{suffix}']

    return collection


def create_dose_tmp_2_df(suffix, unique):
    parameters = {
        f'dose_small_tmp_{suffix}': unique,
        f'duration_{suffix}': [1, 2, 3, 5, 8, 13, 21],
    }

    dtypes = {
        f'dose_small_tmp_{suffix}': 'float64',
        f'duration_{suffix}': 'int8',
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_tmp_2_{suffix}'] = \
        collection[f'dose_small_tmp_{suffix}'] * \
        collection[f'duration_{suffix}']

    collection[f'dose_tmp_2_{suffix}'] = \
        collection[f'dose_tmp_2_{suffix}'].mask(
        collection[f'dose_tmp_2_{suffix}'] > 1,
        1
    )

    return collection


def create_dose_tmp_3_df(suffix, unique):
    # TODO: create dfs for num_infected, taking in the number of people times
    # the infection rate
    parameters = {
        f'dose_tmp_2_{suffix}': unique,
        f'num_infected_{suffix}': range(100),
    }

    dtypes = {
        f'dose_tmp_2_{suffix}': 'float64',
        f'num_infected_{suffix}': 'int8'
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_tmp_3_{suffix}'] = \
        collection[f'dose_tmp_2_{suffix}'] * \
        collection[f'num_infected_{suffix}']

    collection[f'dose_tmp_3_{suffix}'] = \
        collection[f'dose_tmp_3_{suffix}'].mask(
        collection[f'dose_tmp_3_{suffix}'] > 1,
        1
    )

    return collection


def create_dose_tmp_4_df(suffix, unique):
    # TODO: create dfs for num_infected, taking in the number of people times
    # the infection rate
    parameters = {
        f'dose_tmp_3_{suffix}': unique,
        f'emission_at_resting_{suffix}':
            [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100],
    }

    dtypes = {
        f'dose_tmp_3_{suffix}': 'float64',
        f'emission_at_resting_{suffix}': 'int16'
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_tmp_4_{suffix}'] = \
        collection[f'dose_tmp_3_{suffix}'] * \
        collection[f'emission_at_resting_{suffix}']

    collection[f'dose_tmp_4_{suffix}'] = \
        collection[f'dose_tmp_4_{suffix}'].mask(
        collection[f'dose_tmp_4_{suffix}'] > 1,
        1
    )

    return collection


def create_dose_tmp_5_df(suffix, unique, rounding=5):
    # TODO: create dfs for num_infected, taking in the number of people times
    # the infection rate
    parameters = {
        f'dose_tmp_4_{suffix}': unique,
        f'emission_factor_{suffix}':
            [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100, 200],
    }

    dtypes = {
        f'dose_tmp_4_{suffix}': 'float64',
        f'emission_factor_{suffix}':
            'int16',
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_tmp_5_{suffix}'] = \
        collection[f'dose_tmp_4_{suffix}'] * \
        collection[f'emission_factor_{suffix}']

    # Do rounding to a good-enough precision so that the merge with the
    # probability of infection is quick
    collection[f'dose_tmp_5_{suffix}'] = \
        collection[f'dose_tmp_5_{suffix}'].mask(
        collection[f'dose_tmp_5_{suffix}'] > 1,
        1
    ).round(rounding)

    return collection


def create_dose_df(suffix, unique):
    # TODO: create dfs for num_infected, taking in the number of people times
    # the infection rate
    parameters = {
        f'dose_tmp_5_{suffix}': unique,
        f'inhalation_factor_{suffix}':
            [1, 2, 5, 10],
    }

    dtypes = {
        f'dose_tmp_5_{suffix}': 'float64',
        f'inhalation_factor_{suffix}':
            'int8',
    }

    collection = mega_join_cross_product(parameters, dtypes)

    collection[f'dose_{suffix}'] = \
        collection[f'dose_tmp_5_{suffix}'] * \
        collection[f'inhalation_factor_{suffix}']

    collection[f'dose_{suffix}'] = \
        collection[f'dose_{suffix}'].mask(
        collection[f'dose_{suffix}'] > 1,
        1
    )

    return collection


def add(df, key_1, key_2):
    return df[key_1] + df[key_2]


def mult(df, key_1, key_2):
    return df[key_1] * df[key_2]


def div(df, key_1, key_2):
    return df[key_1] / df[key_2]


def create_tmp_df(parameters, dtypes, new_key, func):
    collection = mega_join_cross_product(parameters, dtypes)
    keys = list(parameters.keys())
    key_1 = keys[0]
    key_2 = keys[1]

    collection[new_key] = func(collection, key_1, key_2)

    return collection


def create_infectious_proba_tmp_1_df(suffix, rounding=5):
    parameters = {
        f'ratio_reported_cases_per_pop_{suffix}':
            np.arange(0.0, 0.0005, 0.00001).round(rounding),
        f'reported_frac_given_infectious_{suffix}':
            np.arange(0.1, 0.9, 0.1).round(rounding)
    }

    dtypes = {
        f'ratio_reported_cases_per_pop_{suffix}':
            'float64',
        f'reported_frac_given_infectious_{suffix}': 'float64'
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=f'infectious_proba_tmp_1_{suffix}',
        func=div
    )

    return df


def create_infectious_proba_df(suffix, unique):
    parameters = {
        'infectivity_period_length':
            list(range(1, 14)),
        f'infectious_proba_tmp_1_{suffix}':
            unique
    }

    dtypes = {
        'infectivity_period_length': 'int8',
        f'infectious_proba_tmp_1_{suffix}':
            'float64',
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=f'infectious_proba_{suffix}',
        func=mult
    )

    return df


def create_num_infected_tmp_df(suffix, unique):
    parameters = {
        f'num_people_seen_{suffix}': list(range(0, 100)),
        f'infectious_proba_{suffix}': unique
    }

    dtypes = {
        f'num_people_seen_{suffix}': 'int8',
        f'infectious_proba_{suffix}':
            'float64',
    }

    return mega_join_cross_product(parameters, dtypes)


def create_num_infected_df(suffix, tmp_df):
    tmp_df_2 = pd.DataFrame(
        {
            'tmp': 1,
            f'k_{suffix}': list(range(0, 100))
        }
    )

    tmp_df_1 = tmp_df.copy()

    tmp_df_1['tmp'] = 1
    merged = tmp_df_2.merge(tmp_df_1, on='tmp').drop(columns=['tmp'])

    binomial = merged[
        merged[f'k_{suffix}'] <= merged[f'num_people_seen_{suffix}']
    ].copy()

    binomial.loc[:, 'value'] = binom.pmf(
        k=binomial[f'k_{suffix}'],
        n=binomial[f'num_people_seen_{suffix}'],
        p=binomial[f'infectious_proba_{suffix}']
    )

    binomial[f'num_infected_{suffix}'] = binomial[f'k_{suffix}']

    return binomial


def create_num_infected(suffix, bayesian_network):
    infectious_proba_tmp_1_df = create_infectious_proba_tmp_1_df(
        suffix
    )

    infectious_proba_df = create_infectious_proba_df(
        suffix,
        infectious_proba_tmp_1_df[
            f'infectious_proba_tmp_1_{suffix}'
        ].unique()
    )

    tmp_df = create_num_infected_tmp_df(
        suffix,
        infectious_proba_df[f'infectious_proba_{suffix}'].unique()
    )

    num_infected = create_num_infected_df(
        suffix,
        tmp_df
    )

    mappings = {
        f'infectious_proba_tmp_1_{suffix}': infectious_proba_tmp_1_df,
        f'infectious_proba_{suffix}': infectious_proba_df,
        f'num_infected_{suffix}': num_infected
    }

    for outcome_name, df in mappings.items():
        add_edge_to_bn(
            bayesian_network,
            df=df,
            outcome_var=outcome_name,
            storage_folder=None
        )


def create_dose(suffix, bayesian_network):
    """
    Create
    """
    # Emission at resting
    #     ep0: 1 to 50 (1,2,4,8,16,32,64)
    # Emission factor enhancement
    #     r_e: 1-100ish (1,2,4,8,16,32,64,128)
    #
    # Inhalation factor enhancement
    #     r_b: (1,2,5,10)
    #
    # Basic breathing rates of susceptible b_i
    #     b_0i: 0.288 m^3 / h
    #
    # Infected exhalation Filtration eff factor:
    #     f_e: 0.3, 0.5, 0.9, 0.99
    #
    # Susceptible inhalation Filtration eff factor:
    #     f_i: 0.3, 0.5, 0.9, 0.99
    #
    # volume (m^3)
    #     v: 20, 40, 80, 160, 320, 640, 1280, 2560, 5120
    #
    # ventilation parameter for ventilation / air cleaning: 0.1 to 30
    #     l: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8, 16, 20,
    #     30

    dose_small_tmp_df = create_dose_small_tmp_df(suffix)
    dose_tmp_2_df = create_dose_tmp_2_df(
        suffix,
        dose_small_tmp_df[f'dose_small_tmp_{suffix}'].unique()
    )
    dose_tmp_3_df = create_dose_tmp_3_df(
        suffix,
        dose_tmp_2_df[f'dose_tmp_2_{suffix}'].unique()
    )
    dose_tmp_4_df = create_dose_tmp_4_df(
        suffix,
        dose_tmp_3_df[f'dose_tmp_3_{suffix}'].unique()
    )
    dose_tmp_5_df = create_dose_tmp_5_df(
        suffix,
        dose_tmp_4_df[f'dose_tmp_4_{suffix}'].unique()
    )

    dose_df = create_dose_df(
        suffix,
        dose_tmp_5_df[f'dose_tmp_5_{suffix}'].unique()
    )

    mappings = {
        f'dose_small_tmp_{suffix}': dose_small_tmp_df,
        f'dose_tmp_2_{suffix}': dose_tmp_2_df,
        f'dose_tmp_3_{suffix}': dose_tmp_3_df,
        f'dose_tmp_4_{suffix}': dose_tmp_4_df,
        f'dose_tmp_5_{suffix}': dose_tmp_5_df,
        f'dose_{suffix}': dose_df
    }

    for outcome_name, df in mappings.items():
        add_edge_to_bn(
            bayesian_network,
            df=df,
            outcome_var=outcome_name,
            storage_folder=None
        )


def create_volume_ventilation_df(suffix, new_key):
    """
    Multiply the volume (m^3) and the ventilation (h^-1).

    Parameters:
        suffix: string
        new_key: string
            The name that we'll assign to the new column.
    """
    parameters = {
        f'ventilation_{suffix}': [0.1, 1, 2, 4, 8, 16, 20, 500],
        f'volume_{suffix}':
            [10, 20, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880],
    }

    dtypes = {
        f'ventilation_{suffix}': 'float64',
        f'volume_{suffix}':
            'int16',
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=new_key,
        func=mult
    )

    return df


def divide_1_by(divisor_unique, divisor_name, new_key):
    """
    Divide 1 by some values.

    Parameters:
        divisor_unique: list
            List of unique values that we'll divide by.

        divisor_name: string
            The current name of the divisor.

        new_key: string
            The name that we'll assign to the new column.

    Returns: pd.DataFrame
    """
    parameters = {
        'one': [1.0],
        divisor_name: divisor_unique
    }

    dtypes = {
        'one': 'float64',
        divisor_name: 'float64',
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=new_key,
        func=div
    )

    return df.drop(columns=['one'])


def multiply_by(
    factor_1_unique,
    factor_1_name,
    factor_2_unique,
    factor_2_name,
    new_key,
    factor_1_dtype='float64',
    factor_2_dtype='float64',
):
    """
    Divide 1 by some values.

    Parameters:
        divisor_unique: list
            List of unique values that we'll divide by.

        divisor_name: string
            The current name of the divisor.

        new_key: string
            The name that we'll assign to the new column.

    Returns: pd.DataFrame
    """
    parameters = {
        factor_1_name: factor_1_unique,
        factor_2_name: factor_2_unique
    }

    dtypes = {
        factor_1_name: factor_1_dtype,
        factor_2_name: factor_2_dtype,
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=new_key,
        func=mult
    )

    return df


def multiply_mask_inhalation(unique):
    """
    Multiply the volume (m^3) and the ventilation (h^-1).

    Parameters:
        suffix: string
        new_key: string
            The name that we'll assign to the new column.
    """
    parameters = {
        f'ventilation_{suffix}': [0.1, 1, 2, 4, 8, 16, 20, 40, 80, 160, 240, 480],
        f'volume_{suffix}':
            [10, 20, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880],
    }

    dtypes = {
        f'ventilation_{suffix}': 'float64',
        f'volume_{suffix}':
            'int16',
    }

    df = create_tmp_df(
        parameters,
        dtypes,
        new_key=new_key,
        func=mult
    )

    return df


def cap(df, key, maximum):
    df[key] = df[key].mask(
        df[key] > maximum,
        maximum
    )


def create_room_event(event_suffix, bayesian_network):
    """
    Create parameters shared by people in a room, such as the volume of the
    room and the ventilation rate.

    Parameters:

    Returns: tuple(string, pd.DataFrame)
        First argument is a string that represents the key of doing 1 / (volume
        * ventilation)

        Second argument is a dataframe representing an intermediary conditional
        probability table for calculating doses inhaled by an individual.

    """
    new_key_1 = f'volume_vent_mult_{event_suffix}'

    volume_ventilation_df = create_volume_ventilation_df(
        suffix=event_suffix,
        new_key=new_key_1
    )

    new_key_2 = f'volume_vent_div_{event_suffix}'

    tmp_2_df = divide_1_by(
        divisor_unique=volume_ventilation_df[new_key_1],
        divisor_name=new_key_1,
        new_key=new_key_2
    )

    keys = [
        new_key_1,
        new_key_2,
    ]

    dfs = [
        volume_ventilation_df,
        tmp_2_df,
    ]

    for key, df in zip(keys, dfs):
        add_edge_to_bn(
            bayesian_network,
            df=df,
            outcome_var=key,
            storage_folder=None
        )

    return new_key_2, tmp_2_df


def create_activity_specific_breathing_rate_df(
    person_breathing_in,
    time,
    event,
    breathing_rate_key,
    rounding=5
):
    """
    Generate breathing rates taking into account age and activity intensity.

    Parameters:
        person_breathing_in: string
            E.g. "person 1"
        time: string
            E.g. "2022-01"
        event: string
            E.g. "work", "party"

    Returns: tuple(pd.DataFrame)
    """
    age_key = f'age_{person_breathing_in}'

    person_time_event_index = index_name(
        time,
        event,
        person_breathing_in
    )

    activity_key = f'activity_{person_time_event_index}'

    # below is in cubic meters per minute
    keys = {
        'Sleep or Nap': {
            'Birth to <1':
                np.arange(0.003, 0.0046, 0.0001).round(rounding),
            '1 to <2':
                np.arange(0.0045, 0.0064, 0.0001).round(rounding),
            '2 to <3':
                np.arange(0.0046, 0.0064, 0.0001).round(rounding),
            '3 to <6':
                np.arange(0.0043, 0.0058, 0.0001).round(rounding),
            '6 to <11':
                np.arange(0.0045, 0.0063, 0.0001).round(rounding),
            '11 to <16':
                np.arange(0.0050, 0.0074, 0.0001).round(rounding),
            '16 to <21':
                np.arange(0.0049, 0.0071, 0.0001).round(rounding),
            '21 to <31':
                np.arange(0.0043, 0.0065, 0.0001).round(rounding),
            '31 to <41':
                np.arange(0.0046, 0.0066, 0.0001).round(rounding),
            '41 to <51':
                np.arange(0.0050, 0.0071, 0.0001).round(rounding),
            '51 to <61':
                np.arange(0.0052, 0.0075, 0.0001).round(rounding),
            '61 to <71':
                np.arange(0.0052, 0.0072, 0.0001).round(rounding),
            '71 to <81':
                np.arange(0.0053, 0.0072, 0.0001).round(rounding),
            '>=81':
                np.arange(0.0052, 0.0070, 0.0001).round(rounding),
        },
        'Sedentary/Passive': {
            'Birth to <1':
                np.arange(0.0031, 0.0047, 0.0001).round(rounding),
            '1 to <2':
                np.arange(0.0047, 0.0066, 0.0001).round(rounding),
            '2 to <3':
                np.arange(0.0048, 0.0065, 0.0001).round(rounding),
            '3 to <6':
                np.arange(0.0045, 0.0058, 0.0001).round(rounding),
            '6 to <11':
                np.arange(0.0048, 0.0064, 0.0001).round(rounding),
            '11 to <16':
                np.arange(0.0054, 0.0075, 0.0001).round(rounding),
            '16 to <21':
                np.arange(0.0053, 0.0072, 0.0001).round(4),
            '21 to <31':
                np.arange(0.0042, 0.0065, 0.0001).round(rounding),
            '31 to <41':
                np.arange(0.0043, 0.0066, 0.0001).round(rounding),
            '41 to <51':
                np.arange(0.0048, 0.0070, 0.0001).round(rounding),
            '51 to <61':
                np.arange(0.0050, 0.0073, 0.0001).round(rounding),
            '61 to <71':
                np.arange(0.0049, 0.0070, 0.0001).round(rounding),
            '71 to <81':
                np.arange(0.0050, 0.0072, 0.0001).round(rounding),
            '>=81':
                np.arange(0.0049, 0.0070, 0.0001).round(rounding),
        },
        'Light Intensity': {
            'Birth to <1':
                np.arange(0.0076, 0.0011, 0.0001).round(rounding),
            '1 to <2':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '2 to <3':
                np.arange(0.011, 0.014, 0.0001).round(rounding),
            '3 to <6':
                np.arange(0.011, 0.015, 0.0001).round(rounding),
            '6 to <11':
                np.arange(0.011, 0.015, 0.0001).round(rounding),
            '11 to <16':
                np.arange(0.013, 0.017, 0.0001).round(rounding),
            '16 to <21':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '21 to <31':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '31 to <41':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '41 to <51':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '51 to <61':
                np.arange(0.013, 0.017, 0.0001).round(rounding),
            '61 to <71':
                np.arange(0.012, 0.016, 0.0001).round(rounding),
            '71 to <81':
                np.arange(0.012, 0.015, 0.0001).round(rounding),
            '>=81':
                np.arange(0.012, 0.015, 0.0001).round(rounding),
        },
        'Moderate Intensity': {
            'Birth to <1':
                np.arange(0.014, 0.016, 0.0001).round(rounding),
            '1 to <2':
                np.arange(0.021, 0.029, 0.0001).round(rounding),
            '2 to <3':
                np.arange(0.021, 0.029, 0.0001).round(rounding),
            '3 to <6':
                np.arange(0.021, 0.029, 0.0001).round(rounding),
            '6 to <11':
                np.arange(0.022, 0.029, 0.0001).round(rounding),
            '11 to <16':
                np.arange(0.025, 0.034, 0.0001).round(rounding),
            '16 to <21':
                np.arange(0.026, 0.037, 0.0001).round(rounding),
            '21 to <31':
                np.arange(0.026, 0.038, 0.0001).round(rounding),
            '31 to <41':
                np.arange(0.026, 0.038, 0.0001).round(rounding),
            '41 to <51':
                np.arange(0.028, 0.039, 0.0001).round(rounding),
            '51 to <61':
                np.arange(0.029, 0.040, 0.0001).round(rounding),
            '61 to <71':
                np.arange(0.026, 0.034, 0.0001).round(rounding),
            '71 to <81':
                np.arange(0.025, 0.032, 0.0001).round(rounding),
            '>=81':
                np.arange(0.025, 0.031, 0.0001).round(rounding),
        },
        'High Intensity': {
            'Birth to <1':
                np.arange(0.026, 0.041, 0.0001).round(rounding),
            '1 to <2':
                np.arange(0.038, 0.052, 0.0001).round(rounding),
            '2 to <3':
                np.arange(0.039, 0.052, 0.0001).round(rounding),
            '3 to <6':
                np.arange(0.039, 0.053, 0.0001).round(rounding),
            '6 to <11':
                np.arange(0.042, 0.059, 0.0001).round(rounding),
            '11 to <16':
                np.arange(0.049, 0.070, 0.0001).round(rounding),
            '16 to <21':
                np.arange(0.049, 0.073, 0.0001).round(rounding),
            '21 to <31':
                np.arange(0.050, 0.076, 0.0001).round(rounding),
            '31 to <41':
                np.arange(0.049, 0.072, 0.0001).round(rounding),
            '41 to <51':
                np.arange(0.052, 0.076, 0.0001).round(rounding),
            '51 to <61':
                np.arange(0.053, 0.078, 0.0001).round(rounding),
            '61 to <71':
                np.arange(0.047, 0.066, 0.0001).round(rounding),
            '71 to <81':
                np.arange(0.047, 0.065, 0.0001).round(rounding),
            '>=81':
                np.arange(0.048, 0.068, 0.0001).round(rounding),
        }
    }

    collection = []
    for activity, ages in keys.items():
        for age, rng in ages.items():
            df = pd.DataFrame(
                {
                    breathing_rate_key: rng,
                    activity_key: [
                        activity for _ in range(len(rng))
                    ],
                    age_key: [
                        age for _ in range(len(rng))
                    ],
                    'value': 1
                }
            )

            collection.append(df)

    cpt_df = pd.concat(collection)

    # So that the result is in cubic meters per hour
    cpt_df[breathing_rate_key] = cpt_df[breathing_rate_key] * 60
    return cpt_df


def create_dose_pair_between(
    potential_infector,
    potential_infectee,
    inhalation_factor_for_event_person_name,
    inhalation_factor_for_event_person_unique,
    exhalation_factor_key,
    time,
    event,
    volume_vent_key,
    volume_vent_df,
    bayesian_network,
    rounding=5,
    prefix='dose_tmp',
    cap_at=10,
):

    potential_infector_event_suffix = index_name(
        time,
        event,
        potential_infector
    )

    potential_infector_day_suffix = index_name(
        time,
        potential_infector
    )

    potential_infectee_event_suffix = index_name(
        time,
        event,
        potential_infectee
    )

    pair_suffix = index_name(
        time,
        event,
        potential_infector,
        potential_infectee
    )

    new_key_3 = f'{prefix}_3_{pair_suffix}'
    tmp_3_df = multiply_by(
        factor_1_unique=volume_vent_df[volume_vent_key].unique(),
        factor_1_name=volume_vent_key,
        factor_2_unique=[1, 0.7, 0.4, 0.10, 0.01],
        factor_2_name=f'mask_{potential_infector_event_suffix}',
        new_key=new_key_3
    )
    cap(df=tmp_3_df, key=new_key_3, maximum=cap_at)
    tmp_3_df[new_key_3] = tmp_3_df[new_key_3].round(rounding)

    new_key_4 = f'{prefix}_4_{pair_suffix}'
    tmp_4_df = multiply_by(
        factor_1_unique=tmp_3_df[new_key_3].unique(),
        factor_1_name=new_key_3,
        factor_2_unique=[1, 0.7, 0.4, 0.10, 0.01],
        factor_2_name=f'mask_{potential_infectee_event_suffix}',
        new_key=new_key_4
    )
    cap(df=tmp_4_df, key=new_key_4, maximum=cap_at)
    tmp_4_df[new_key_4] = tmp_4_df[new_key_4].round(rounding)
    new_key_5 = f'{prefix}_5_{pair_suffix}'

    tmp_5_df = multiply_by(
        factor_1_unique=inhalation_factor_for_event_person_unique,
        factor_1_name=inhalation_factor_for_event_person_name,
        factor_2_unique=tmp_4_df[new_key_4].unique(),
        factor_2_name=new_key_4,
        new_key=new_key_5
    )
    cap(df=tmp_5_df, key=new_key_5, maximum=cap_at)
    tmp_5_df[new_key_5] = tmp_5_df[new_key_5].round(rounding)

    new_key_7 = f'{prefix}_7_{pair_suffix}'
    tmp_7_df = multiply_by(
            factor_1_unique=tmp_5_df[new_key_5].unique(),
            factor_1_name=new_key_5,
            factor_2_unique=[
                1, 4.7, 30.3, 1.2, 5.7, 32.6, 2.8, 13.2, 85, 6.8, 31.6, 204
            ],
            factor_2_name=f'exhalation_factor_{potential_infector_event_suffix}',
            new_key=new_key_7
            )

    cap(df=tmp_7_df, key=new_key_7, maximum=cap_at)
    tmp_7_df[new_key_7] = tmp_7_df[new_key_7].round(rounding)

    new_key_8 = f'{prefix}_8_{pair_suffix}'
    tmp_8_df = multiply_by(
            factor_1_unique=tmp_7_df[new_key_7].unique(),
            factor_1_name=new_key_7,
            factor_2_unique=[
                1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100, 200],
            factor_2_name=f'quanta_{potential_infector_day_suffix}',
            new_key=new_key_8
            )

    cap(df=tmp_8_df, key=new_key_8, maximum=cap_at)
    tmp_8_df[new_key_8] = tmp_8_df[new_key_8].round(rounding)

    new_key_9 = f'{prefix}_9_{pair_suffix}'
    tmp_9_df = multiply_by(
            factor_1_unique=tmp_8_df[new_key_8].unique(),
            factor_1_name=new_key_8,
            factor_2_unique=[
                0.25, 0.5, 1, 2, 3, 5, 8, 13, 21
            ],
            factor_2_name=f'duration_{pair_suffix}',
            new_key=new_key_9
            )

    keys = [
            new_key_3,
            new_key_4,
            new_key_5,
            new_key_7,
            new_key_8,
            new_key_9,
            ]

    dfs = [
        tmp_3_df,
        tmp_4_df,
        tmp_5_df,
        tmp_7_df,
        tmp_8_df,
        tmp_9_df,
    ]

    add_dfs_to_bn(
        bayesian_network,
        dfs,
        keys
    )


def add_dfs_to_bn(
    bayesian_network,
    dfs,
    keys,
    storage_folder=None
):
    for key, df in zip(keys, dfs):
        add_edge_to_bn(
            bayesian_network,
            df=df,
            outcome_var=key,
            storage_folder=None
        )


def create_activity_exhalation(
    time,
    event,
    person,
    bayesian_network,
    storage_folder=None
):
    suffix = index_name(time, event, person)
    inhalation_rate_key = \
        f'inhalation_rate_{suffix}'

    exhalation_factor_key = \
        f'exhalation_factor_{suffix}'

    activity_exhalation_df = pd.DataFrame({
        f'activity_exhalation_{suffix}': [
            "Resting – Oral breathing",
            "Resting – Speaking",
            "Resting – Loudly speaking",
            "Standing – Oral breathing",
            "Standing – Speaking",
            "Standing – Loudly speaking",
            "Light exercise – Oral breathing",
            "Light exercise – Speaking",
            "Light exercise – Loudly speaking",
            "Heavy exercise – Oral breathing",
            "Heavy exercise – Speaking",
            "Heavy exercise – Loudly speaking"
        ],
        exhalation_factor_key: [
            1, 4.7, 30.3, 1.2, 5.7, 32.6, 2.8, 13.2, 85, 6.8, 31.6, 204
        ],
        'value': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    })

    keys = [exhalation_factor_key]
    dfs = [activity_exhalation_df]

    add_dfs_to_bn(
        bayesian_network,
        dfs,
        keys,
        storage_folder=storage_folder
    )

    return {
        'inhalation_rate_key': inhalation_rate_key,
        'exhalation_factor_key': exhalation_factor_key
    }


def create_doses(
    people,
    time,
    event,
    volume_vent_key,
    volume_vent_df,
    bayesian_network,
    storage_folder=None,
    prefix='dose_tmp'
):

    for p1 in people:
        activity_exh_dict = create_activity_exhalation(
            time,
            event,
            p1,
            bayesian_network,
            storage_folder=storage_folder
        )

        exhalation_factor_key = activity_exh_dict['exhalation_factor_key']

        for p2 in people:
            if p1 == p2:
                continue

            inhalation_rate_key = \
                f'inhalation_rate_{index_name(time, event, p2)}'

            tmp_5a_df = create_activity_specific_breathing_rate_df(
                p2,
                time,
                event,
                inhalation_rate_key
            )

            unique_values = tmp_5a_df[inhalation_rate_key].unique()
            create_dose_pair_between(
                potential_infector=p1,
                potential_infectee=p2,
                time=time,
                event=event,
                inhalation_factor_for_event_person_unique=unique_values,
                inhalation_factor_for_event_person_name=inhalation_rate_key,
                exhalation_factor_key=exhalation_factor_key,
                volume_vent_key=volume_vent_key,
                volume_vent_df=volume_vent_df,
                bayesian_network=bayesian_network,
                prefix=prefix
            )

    sum_up_doses_of_people(
        people,
        time,
        event,
        bayesian_network,
        prefix=f'{prefix}_9',
        rounding=2,
        storage_folder=storage_folder
    )


def sum_up_doses_of_people(
    people,
    time,
    event,
    bayesian_network,
    prefix="tmp_9",
    rounding=5,
    storage_folder=None
):
    """
    Add the doses for a person (received from other individuals)
    Creates dataframes that only have two parent.

    Parameters:

        people: list[string]
            The people indi
    """
    for receiver in people:

        key_1 = None
        key_2 = None

        for other_index, giver in enumerate(people):
            if receiver == giver:
                continue

            if key_1 is None:
                key_1 = f"{prefix}_({time}, {event}, {giver}, {receiver})"
                continue

            key_2 = f"{prefix}_({time}, {event}, {giver}, {receiver})"

            # Below, we have two keys available

            parameters = {
                key_1: np.arange(0.0, 2.0, 0.01).round(rounding),
                key_2: np.arange(0.0, 2.0, 0.01).round(rounding),
            }

            dtypes = {
                key_1: 'float64',
                key_2: 'float64',
            }

            new_key = f"dose_tmp_({time}, {receiver}, {other_index})"

            df = create_tmp_df(
                parameters,
                dtypes,
                new_key=new_key,
                func=add
            )

            cap(df=df, key=new_key, maximum=2)
            df[new_key] = df[new_key].round(rounding)

            add_edge_to_bn(
                bayesian_network,
                df=df,
                outcome_var=new_key,
                storage_folder=storage_folder
            )

            key_1 = new_key


def create_dose_pair(
    person_1,
    person_2,
    time,
    event,
    volume_vent_key,
    volume_vent_df,
    bayesian_network,
    prefix
):
    create_dose_pair_between(
        potential_infector=person_1,
        potential_infectee=person_2,
        time=time,
        event=event,
        volume_vent_key=volume_vent_key,
        volume_vent_df=volume_vent_df,
        bayesian_network=bayesian_network,
        prefix=prefix
    )

    create_dose_pair_between(
        potential_infector=person_2,
        potential_infectee=person_1,
        time=time,
        event=event,
        volume_vent_key=volume_vent_key,
        volume_vent_df=volume_vent_df,
        bayesian_network=bayesian_network,
        prefix=prefix
    )


def create_dose_pair_tmp_df_1(suffix):
    parameters = {
            f'mask_exhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
            f'mask_inhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
            }

    dtypes = {
            f'mask_exhalation_factor_{suffix}': 'float64',
            f'mask_inhalation_factor_{suffix}': 'float64',
            }

    df = create_tmp_df(
            suffix,
            parameters,
            dtypes,
            new_key=f'tmp_1_{suffix}',
            func=mult
            )


def create_dose_pair_tmp_df_2(suffix):
    parameters = {
            f'mask_exhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
            f'mask_inhalation_factor_{suffix}': [1, 0.7, 0.4, 0.10, 0.01],
            }

    dtypes = {
            f'mask_exhalation_factor_{suffix}': 'float64',
            f'mask_inhalation_factor_{suffix}': 'float64',
            }

    df = create_tmp_df(
            suffix,
            parameters,
            dtypes,
            new_key=f'tmp_1_{suffix}',
            func=mult
            )

    return df


def create_dose_pair_tmp_df_2(suffix, unique):
    parameters = {
        f'tmp_1_{suffix}': unique,
        f'volume_{suffix}':
        [10, 20, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880],
    }

    dtypes = {
        f'tmp_1_{suffix}': 'float64',
        f'volume_{suffix}':
        'int16',
    }

    df = create_tmp_df(
        suffix,
        parameters,
        dtypes,
        new_key=f'tmp_2_{suffix}',
        func=div
    )

    return df


def create_dose_pair_tmp_df_3(suffix, unique):
    parameters = {
        f'tmp_2_{suffix}': unique,
        f'ventilation_factor_{suffix}':
            [0.1, 0.5, 1, 2, 3, 5, 8, 13, 21, 34],
    }

    dtypes = {
        f'tmp_2_{suffix}': 'float64',
        f'ventilation_factor_{suffix}':
            'float64',
    }

    df = create_tmp_df(
        suffix,
        parameters,
        dtypes,
        new_key=f'tmp_3_{suffix}',
        func=div
    )

    return df


def create_dose_pair_tmp_df_4(suffix, unique):
    parameters = {
        f'tmp_3_{suffix}': unique,
        f'inhalation_at_rest_{suffix}': [0.288]
    }

    dtypes = {
        f'tmp_3_{suffix}': 'float64',
        f'inhalation_at_rest_{suffix}':
            'float64',
    }

    df = create_tmp_df(
        suffix,
        parameters,
        dtypes,
        new_key=f'tmp_4_{suffix}',
        func=div
    )

    return df


def create_dose_pair_tmp_df_5(suffix, unique):
    parameters = {
        f'tmp_4_{suffix}': unique,
        f'inhalation_at_rest_{suffix}': [0.288]
    }

    dtypes = {
        f'tmp_3_{suffix}': 'float64',
        f'inhalation_at_rest_{suffix}':
            'float64',
    }

    df = create_tmp_df(
        suffix,
        parameters,
        dtypes,
        new_key=f'tmp_4_{suffix}',
        func=div
    )

    return df


# def create_dose_pair_tmp_df_2(suffix, unique):
    # parameters = {
        # f'tmp_2_{suffix}': unique,
        # f'volume_{suffix}':
            # [10, 20, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880],
        # f'ventilation_factor_{suffix}': [0.1, 0.5, 1, 2, 3, 5, 8, 13, 21, 34],
        # f'inhalation_at_rest_{suffix}': [0.288]
    # }
#
    # dtypes = {
        # f'mask_exhalation_factor_{suffix}': 'float64',
        # f'mask_inhalation_factor_{suffix}': 'float64',
        # f'volume_{suffix}':
            # 'int16',
        # f'ventilation_factor_{suffix}': 'float64',
        # f'inhalation_at_rest_{suffix}': 'float64'
    # }
#
    # collection = mega_join_cross_product(parameters, dtypes)
#
    # collection[f'dose_small_tmp_{suffix}'] = \
        # collection[f'mask_exhalation_factor_{suffix}'] * \
        # collection[f'mask_inhalation_factor_{suffix}'] * \
        # collection[f'inhalation_at_rest_{suffix}'] / \
        # collection[f'volume_{suffix}'] / \
        # collection[f'ventilation_factor_{suffix}']
#
    # return collection


def create_virus_levels_tmp_df(suffix):
    """
    Create virus levels using days-since-infection, assuming that it is
    negative-binomially distributed.
    """
    dsi_suffixed = f'dsi_{suffix}'
    n_list = [10, 11, 12, 13, 14, 15, 16]
    p_list = [0.6]

    dfs = []

    for n in n_list:
        for p in p_list:
            dsi = np.arange(
                nbinom.ppf(0.001, n, p),
                nbinom.ppf(0.999, n, p)
            )

            df = pd.DataFrame({
                dsi_suffixed: dsi,
            })

            df[f'virus_levels_{suffix}'] = \
                nbinom.pmf(dsi, n, p)

            df['virus_binom_n'] = n
            df['virus_binom_p'] = p
            df['dsi_{suffix}'] = dsi

            df['value'] = 1.0

            dfs.append(df)

    return pd.concat(dfs)


def create_virus_levels_tmp(suffix, bayesian_network, storage_folder=None):
    df = create_virus_levels_tmp_df(suffix)
    add_edge_to_bn(
        bayesian_network,
        df,
        outcome_var=[f'virus_levels_{suffix}'],
        storage_folder=storage_folder
    )

    return df


def create_pair_dose(
    from_individual,
    to_individual,
    time,
    location
):
    """
    Create conditional probability tables representing dosage received by
    "to_individual" from "from_individual".
    """


def generate_household(self):
    pass


if __name__ == '__main__':
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())

    create_dose(
        suffix=index_name('edderic', '1', 'work'),
        bayesian_network=bayesian_network
    )

    create_num_infected(
        suffix=index_name('edderic', '1', 'work'),
        bayesian_network=bayesian_network
    )
