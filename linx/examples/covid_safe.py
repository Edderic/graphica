"""
CovidSafe: A Risk Estimation to make social gatherings safer.
"""
import graphviz
import numpy as np
import pandas as pd
from scipy.stats import binom


from ..misc import get_tmp_path, clean_tmp

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
    if i is not None and j is None and k is None:
        return f"({i})"
    if i is not None and j is not None and k is None:
        return f"({i}, {j})"
    if i is not None and j is not None and k is not None:
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

    num_days_of_infection = 14
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


def create_dose_tmp_5_df(suffix, unique):
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
    ).round(4)

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


def create_infectious_proba_tmp_1_df(suffix):
    parameters = {
        f'ratio_reported_cases_per_pop_{suffix}':
            np.arange(0.0, 0.0005, 0.00001).round(5),
        f'reported_frac_given_infectious_{suffix}':
            np.arange(0.1, 0.9, 0.1).round(2)
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
    # lambda parameter for ventilation / air cleaning: 0.1 to 30
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
