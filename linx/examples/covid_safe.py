"""
CovidSafe: A Risk Estimation to make social gatherings safer.
"""
import graphviz
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import nbinom



from ..ds import BayesianNetwork as BN, \
    ConditionalProbabilityTable as CPT
from ..data import InMemoryData


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
    dsi_key,
    pre_dsi_key,
    infected_key,
    max_num_days_since_infection=21
):
    """
    Create days of infection for COVID.

    If someone is susceptible (represented as 0) and gets infected, then the
    days-since-infection for the next day gets set to 1. Once the value for
    days-since-infection is between 1 and 27 (inclusive), then
    days-since-infection for the next day
    increase by 1. When we reach day the max_num_days_since_infection,
    we reset back to susceptible.

    Parameters:
        suffix: string
            A string for personalization.

        pre_suffix: string
            A string for personalization.

        max_num_days_since_infection: string
            The number of days to track
    """
    susceptible = 0
    dsi = list(range(susceptible, max_num_days_since_infection))

    parameters = {
        pre_dsi_key: dsi,
        infected_key: [0, 1],
    }

    dtypes = {
        pre_dsi_key: 'int8',
        infected_key: 'int8',
    }

    df = mega_join_cross_product(parameters, dtypes)
    df[dsi_key] = df.groupby(
        [
            infected_key
        ]
    )[pre_dsi_key].shift(-1)

    df[dsi_key].mask(
        (df[pre_dsi_key] == susceptible) &
        (df[infected_key] == 1),
        1,
        inplace=True
    )

    df[dsi_key].mask(
        (df[pre_dsi_key] == susceptible) &
        (df[infected_key] == 0),
        susceptible,
        inplace=True
    )

    df[dsi_key].mask(
        (df[dsi_key].isna()),
        0,
        inplace=True
    )

    df[dsi_key] = df[dsi_key].astype('int8')

    return df


def create_viral_load_n(viral_load_n_key, immunity_key):
    """
    Create parameter for the amount of virus in the system.

    Parameters:
        viral_load_n_key: string
        immunity_key: string
    """

    return pd.DataFrame(
        {
            viral_load_n_key: pd.Series([
                10, 11, 12, 13, 14, 15, 16,
                10, 11, 12, 13, 14, 15, 16,
            ], dtype='int8'),
            immunity_key: pd.Series([
                1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0
            ], dtype='int8'),
            'value': pd.Series([
                0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05,
                0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2,
            ], dtype='float64')
        }
    )


def create_viral_load_p(viral_load_p_key, immunity_key):
    """
    Create parameter for the amount of virus in the system.

    Parameters:
        viral_load_p_key: string
        immunity_key: string
    """

    return pd.DataFrame(
        {
            viral_load_p_key: pd.Series([
                0.6, 0.6
            ], dtype='float64'),
            immunity_key: pd.Series([
                0, 1
            ], dtype='int8'),
            'value': pd.Series([
                1, 1
            ], dtype='float64')
        }
    )


def create_immunity_factor(immunity_key, immunity_factor_key):
    """
    Create immunity factor.

    Parameters:
        immunity_key: string
        immunity_fator_key: string

    Returns: pd.DataFrame
    """
    return pd.DataFrame({
        immunity_key: [
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
        ],
        immunity_factor_key: pd.Series(
            [
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ],
            dtype='float64'
        ),
        'value': pd.Series(
            [
                0.1, 0.1, 0.2, 0.2, 0.2, 0.2,
                0.25, 0.25, 0.2, 0.2, 0.05, 0.05,
            ],
            dtype='float64'
        )
    })


def create_viral_load(
    n_key,
    p_key,
    dsi_key,
    immunity_factor_key,
    viral_load_key,
    unique_n,
    unique_p,
    unique_immunity_factor,
    max_num_days_since_infection=21,
    viral_load_rounding=2
):
    """
    Create viral load curve.

    Parameters:
        unique_n: list[integer]

        unique_p: list[float]

        max_num_days_since_infection: integer
            Max number of days-since-infection.

    Returns: pd.DataFrame
    """
    parameters = {
        n_key: unique_n,
        p_key: unique_p,
        dsi_key: list(range(max_num_days_since_infection)),
        immunity_factor_key: unique_immunity_factor
    }

    dtypes = {
        n_key: 'int8',
        p_key: 'float64',
        dsi_key: 'int8',
        immunity_factor_key: 'float64'
    }

    df = mega_join_cross_product(parameters, dtypes)

    df[viral_load_key] = nbinom.pmf(
        k=df[dsi_key],
        n=df[n_key],
        p=df[p_key]
    )

    df['max'] = df.groupby([n_key, p_key]).transform('max')[viral_load_key]

    # Normalize
    df[viral_load_key] = df[viral_load_key] / df['max']

    df[viral_load_key] = df[viral_load_key] * df[immunity_factor_key]
    df[viral_load_key] = df[viral_load_key].round(viral_load_rounding)

    # 0 represents not infected. If not infected, then viral load should be 0.
    df[viral_load_key].mask(df[dsi_key] == 0, 0.0, inplace=True)

    return df.drop(columns=['max'])


def create_quanta_curve(
    quanta_unique,
    viral_load_key,
    quanta_key,
    viral_load_unique=None,
):
    """
    Produce quanta depending on viral load.

    Parameters:
        viral_load_unique: list[float]
            Currently unused.
        quanta_unique: list[float]

    Returns: pd.DataFrame
    """
    mapping = {
        0.9: 80,
        0.8: 70,
        0.7: 60,
        0.6: 50,
        0.5: 40,
        0.4: 30,
        0.3: 20,
        0.2: 10,
        0.1: 0,
        0.0: 0

    }
    parameters = {
        viral_load_key: list(mapping.keys()),
        quanta_key: quanta_unique
    }

    dtypes = {
        viral_load_key: 'float64',
        quanta_key: 'float64'
    }

    df = mega_join_cross_product(parameters, dtypes)

    for viral_load, quanta in mapping.items():
        df['value'].mask(
            (df[viral_load_key] < viral_load + 0.1)
            & (df[viral_load_key] > viral_load)
            & (df[quanta_key] == quanta),
            1.0,
            inplace=True
        )

    return df


def create_rapid_tests(suffix, rapid_key, viral_load_unique):
    """
    Produce rapid test results depending on viral load.

    Parameters:
        suffix: string
        viral_load_unique: list[float]
        unique_rapid: list

    Returns: pd.DataFrame
    """
    viral_load_key = f'viral_load_{suffix}'

    parameters = {
        viral_load_key: viral_load_unique,
        rapid_key: [0, 1]
    }

    dtypes = {
        viral_load_key: 'float64',
        rapid_key: 'int8'
    }

    df = mega_join_cross_product(parameters, dtypes)
    threshold = 0.5

    df['value'].mask(
        (df[viral_load_key] > threshold) &
        (df[rapid_key] == 1),
        0.95,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] > threshold) &
        (df[rapid_key] == 0),
        0.05,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] <= threshold) &
        (df[rapid_key] == 1),
        0.01,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] <= threshold) &
        (df[rapid_key] == 0),
        0.99,
        inplace=True
    )

    return df


def create_pcr_tests(suffix, pcr_key, viral_load_unique):
    """
    Produce pcr test results depending on viral load.

    Parameters:
        suffix: string
        viral_load_unique: list[float]

    Returns: pd.DataFrame
    """
    viral_load_key = f'viral_load_{suffix}'

    parameters = {
        viral_load_key: viral_load_unique,
        pcr_key: [0, 1]
    }

    dtypes = {
        viral_load_key: 'float64',
        pcr_key: 'int8'
    }

    df = mega_join_cross_product(parameters, dtypes)
    threshold = 0.2

    df['value'].mask(
        (df[viral_load_key] > threshold) &
        (df[pcr_key] == 1),
        0.99,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] > threshold) &
        (df[pcr_key] == 0),
        0.01,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] <= threshold) &
        (df[pcr_key] == 1),
        0.01,
        inplace=True
    )

    df['value'].mask(
        (df[viral_load_key] <= threshold) &
        (df[pcr_key] == 0),
        0.99,
        inplace=True
    )

    return df


def create_symptoms(
    person,
    time,
    start_symp_key,
    end_symp_key,
    start_symp_unique,
    end_symp_unique,
    viral_load_unique,
    max_num_days_since_infection,
):
    """
    Create symptoms for a given person and time.

    Parameters:
        person: string
        time: string

    Returns: pd.DataFrame
    """
    viral_load_threshold = 0.5
    dsi_threshold = 5
    not_infected_dsi = 0

    person_suffix = index_name(person)
    time_person_suffix = index_name(time, person)

    symptomatic_key = f'symptomatic_{person_suffix}'
    viral_load_key = f'viral_load_{time_person_suffix}'
    symptomatic_time_key = f'symptomatic_{time_person_suffix}'
    dsi_key = f'dsi_{time_person_suffix}'

    parameters = {
        symptomatic_key: [0, 1],
        symptomatic_time_key: [0, 1],
        start_symp_key: start_symp_unique,
        end_symp_key: end_symp_unique,
        viral_load_key: viral_load_unique,
        dsi_key: list(range(0, max_num_days_since_infection))
    }

    dtypes = {
        symptomatic_key: 'int8',
        symptomatic_time_key: 'int8',
        start_symp_key: 'int8',
        end_symp_key: 'int8',
        viral_load_key: 'float64',
        dsi_key: 'int8'
    }

    df = mega_join_cross_product(parameters, dtypes)

    # For those who will be symptomatic if sick with COVID
    # If viral load is high
    # OR
    # If day_since_infection is below a threshold AND person has been infected
    # OR
    # the day since infection is between the start and start + end
    get_symptoms = (
        (df[symptomatic_key] == 1) &
        (
            (df[viral_load_key] > viral_load_threshold) |
            (
                (df[dsi_key] < dsi_threshold) &
                (df[dsi_key] != not_infected_dsi)
            ) |
            (
                (
                    df[start_symp_key] <= df[dsi_key]
                ) &
                (
                    df[end_symp_key] + df[start_symp_key] > df[dsi_key]
                )
            )
        )
    )

    # For those who will be "asymptomatic"
    # Most of the time they don't have symptoms
    df['value'].mask(
        (df[symptomatic_time_key] == 1),
        0.05,
        inplace=True
    )

    # For those who will be "asymptomatic"
    # Sometimes they do have symptoms
    df['value'].mask(
        (df[symptomatic_time_key] == 0),
        0.95,
        inplace=True
    )

    # For those who will be "symptomatic"
    # Then they are likely to get symptoms if they meet some criteria
    df['value'].mask(
        (df[symptomatic_time_key] == 1) & get_symptoms,
        0.95,
        inplace=True
    )

    # However, it's still possible not to get symptoms even if some criteria
    # are met.
    df['value'].mask(
        (df[symptomatic_time_key] == 0) & get_symptoms,
        0.05,
        inplace=True
    )

    return df, symptomatic_time_key


def create_inf_dsi_viral_load_measurements(
    person,
    time,
    dose_key,
    bayesian_network,
    time_format,
    viral_load_n_key,
    viral_load_p_key,
    viral_load_n_df,
    viral_load_p_df,
    immunity_key,
    immunity_factor_key,
    immunity_factor_df,
    start_symp_key,
    end_symp_key,
    start_symp_df,
    end_symp_df,
):
    """
    Create probability of infection, days-since-infection, viral load,
    symptoms, pcr & rapid test results.
    """
    prev_time_person_index = index_name(
        (time-timedelta(days=1)).strftime(time_format),
        person
    )

    time_str = time.strftime(time_format)
    max_num_days_since_infection = 21
    time_person_index = index_name(time_str, person)

    infected_key = f'infected_{time_person_index}'
    infection_df = create_infection_from_dose(
        suffix=time_person_index,
        dose_key=dose_key,
        infected_key=infected_key
    )

    dsi_key = f'dsi_{time_person_index}'
    pre_dsi_key = f'dsi_{prev_time_person_index}'

    dsi_df = create_days_since_infection_covid(
        dsi_key,
        pre_dsi_key,
        infected_key,
        max_num_days_since_infection=max_num_days_since_infection
    )

    viral_load_rounding = 2
    viral_load_unique = np.arange(0.0, 1.0, 0.01)\
        .round(viral_load_rounding)

    viral_load_key = f'viral_load_{time_person_index}'

    viral_load_df = create_viral_load(
        n_key=viral_load_n_key,
        p_key=viral_load_p_key,
        dsi_key=dsi_key,
        immunity_factor_key=immunity_factor_key,
        viral_load_key=viral_load_key,
        unique_n=viral_load_n_df[viral_load_n_key].unique(),
        unique_p=viral_load_p_df[viral_load_p_key].unique(),
        unique_immunity_factor=immunity_factor_df[
            immunity_factor_key
        ].unique(),
        max_num_days_since_infection=max_num_days_since_infection,
        viral_load_rounding=viral_load_rounding
    )

    rapid_key = f'rapid_{time_person_index}'
    rapid_test_df = create_rapid_tests(
        suffix=time_person_index,
        rapid_key=rapid_key,
        viral_load_unique=viral_load_df[viral_load_key].unique()
    )

    pcr_key = f'pcr_{time_person_index}'
    pcr_test_df = create_pcr_tests(
        suffix=time_person_index,
        pcr_key=pcr_key,
        viral_load_unique=viral_load_df[viral_load_key].unique()
    )

    symptoms_df, symptomatic_time_key = create_symptoms(
        person,
        time_str,
        start_symp_key=start_symp_key,
        end_symp_key=end_symp_key,
        start_symp_unique=start_symp_df[start_symp_key].unique(),
        end_symp_unique=end_symp_df[end_symp_key].unique(),
        viral_load_unique=viral_load_unique,
        max_num_days_since_infection=max_num_days_since_infection
    )

    quanta_key = f'quanta_{time_person_index}'
    viral_load_key = f'viral_load_{time_person_index}'

    quanta_unique = list(range(0, 90, 10))
    quanta_df = create_quanta_curve(
        viral_load_unique=viral_load_unique,
        quanta_unique=quanta_unique,
        quanta_key=quanta_key,
        viral_load_key=viral_load_key,
    )

    keys = [
        infected_key,
        dsi_key,
        viral_load_key,
        rapid_key,
        pcr_key,
        symptomatic_time_key,
        quanta_key
    ]

    dfs = [
        infection_df,
        dsi_df,
        viral_load_df,
        rapid_test_df,
        pcr_test_df,
        symptoms_df,
        quanta_df
    ]

    for key, df in zip(keys, dfs):
        add_edge_to_bn(
            bayesian_network,
            df=df,
            outcome_var=key,
            storage_folder=None
        )


def create_end_symp(immunity_key, end_symp_key):
    """
    Create the end day of symptoms.

    Parameters:
        immunity_key: string
        end_symp_key: string
    """
    return pd.DataFrame({
        immunity_key: pd.Series(
            [
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
            ],
            dtype='int8'
        ),
        end_symp_key: pd.Series(
            [
                3, 4, 5, 6, 7,
                3, 4, 5, 6, 7,
            ],
            dtype='int8'
        ),
        'value': pd.Series(
            [
                0.05, 0.05, 0.3, 0.3, 0.3,
                0.2, 0.2, 0.2, 0.2, 0.2,
            ],
            dtype='float16'
        )
    })


def create_start_symp(immunity_key, start_symp_key):
    """
    Create the start day of symptoms.

    Parameters:
        immunity_key: string
        start_symp_key: string
    """
    return pd.DataFrame({
        immunity_key: pd.Series(
            [
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
            ],
            dtype='int8'
        ),
        start_symp_key: pd.Series(
            [
                3, 4, 5, 6, 7,
                3, 4, 5, 6, 7,
            ],
            dtype='int8'
        ),
        'value': pd.Series(
            [
                0.05, 0.05, 0.3, 0.3, 0.3,
                0.2, 0.2, 0.2, 0.2, 0.2,
            ],
            dtype='float16'
        )
    })


def mega_join_cross_product(parameters, dtypes):
    """
    Create cross product.

    Parameters:
        parameters: dict
            E.g. {
                'column_1': [1, 2, 3],
                'column_2': [7, 8, 9],
            }
        dtypes: dict
            E.g. {
                'column_1': 'float64',
                'column_2': 'float64',
            }

    Returns: pd.DataFrame
        E.g.
            pd.DataFrame({
                'column_1': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'column_8': [7, 8, 9, 7, 8, 9, 7, 8, 9],
            })
    """
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


def create_volume_ventilation_df(suffix, new_key):
    """
    Multiply the volume (m^3) and the ventilation (h^-1).

    Parameters:
        suffix: string
        new_key: string
            The name that we'll assign to the new column.
    """
    parameters = {
        f'ventilation_{suffix}': [0.5, 1, 2, 4, 8, 10, 16, 20, 500],
        f'volume_{suffix}':
            [10, 20, 30, 40, 60, 100, 160, 260, 420, 680, 1100, 1780, 2880, 10000],
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


def divide_by(
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
        func=div
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
    age_key = f'age_({person_breathing_in})'

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
                np.arange(0.0053, 0.0072, 0.0001).round(rounding),
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


def create_avg_mask_reduction_from_others(suffix):
    avg_mask_reduction_from_others = multiply_by(
        factor_1_unique=np.arange(0.0, 1.1, 0.1).round(1),
        factor_1_name=f'perc_masked_{suffix}',
        factor_2_unique=[
            0, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99
        ],
        factor_2_name=f'mask_quality_{suffix}',
        new_key='mask_product'
    )

    new_key = f'avg_mask_reduction_{suffix}'
    avg_mask_reduction_from_others[new_key] = \
        1.0 - avg_mask_reduction_from_others['mask_product']

    avg_mask_reduction_from_others.drop(
        columns=['mask_product'],
        inplace=True
    )

    avg_mask_reduction_from_others['value'] = 1.0

    return avg_mask_reduction_from_others, new_key


def round_column(df, key, rounding):
    """
    Round numbers to decrease the memory footprint.
    """
    ranges = [
        0,
        1,
        10,
        100,
        1000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000
    ]

    for i, val in enumerate(ranges[:-1]):
        next_val = ranges[i + 1]

        if i == 0:
            actual_rounding = rounding
            power = 0
        else:
            actual_rounding = 0
            power = i - 1

        df[key].mask(
            (df[key] > val) & (df[key] <= next_val),
            (df[key] / 10**(power)).round(actual_rounding) * 10**(power),
            inplace=True
        )


def create_at_least_one_inf(
    time,
    prefix,
    suffix,
    bayesian_network,
    storage_folder=None
):
    num_pos_cases = [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
        10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
        90000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000,
        700_000, 800_000, 900_000, 1_000_000, 2_000_000, 3_000_000,
        4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000,
        9_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000, 70_000_000, 80_000_000, 80_000_000,
        90_000_000, 100_000_000
    ]

    pop_size = num_pos_cases[1:]

    time_suffix = index_name(time)

    new_key_9 = f'{prefix}_pop_ratio_{time_suffix}'
    tmp_9_df = divide_by(
        factor_1_unique=num_pos_cases,
        factor_1_name=f'num_positive_cases_{time_suffix}',
        factor_2_unique=pop_size,
        factor_2_name=f'pop_size_{time_suffix}',
        new_key=new_key_9
    )

    new_key_10 = f'{prefix}_multiplier_1_{time_suffix}'
    tmp_10_df = multiply_by(
        factor_1_unique=tmp_9_df[new_key_9].unique(),
        factor_1_name=new_key_9,
        factor_2_unique=[7, 8, 9, 10],
        factor_2_name=f'num_days_inf_{time_suffix}',
        new_key=new_key_10
    )

    new_key_11 = f'infectious_proba_{time_suffix}'
    tmp_11_df = multiply_by(
        factor_1_unique=tmp_10_df[new_key_10].unique(),
        factor_1_name=new_key_10,
        factor_2_unique=[5, 6, 7, 8, 9, 10],
        factor_2_name=f'unreported_positive_{time_suffix}',
        new_key=new_key_11
    )
    round_column(tmp_11_df, new_key_11, rounding=4)
    tmp_11_df = tmp_11_df[tmp_11_df[new_key_11] < 1]

    parameters = {
        f'num_people_seen_{suffix}': list(range(0, 51)),
        f'infectious_proba_{time_suffix}': tmp_11_df[new_key_11].unique()
    }

    dtypes = {
        f'num_people_seen_{suffix}': 'float64',
        f'infectious_proba_{time_suffix}':
            'float64',
    }

    new_key_12 = f'at_least_one_inf_{suffix}'
    tmp_12_df = mega_join_cross_product(parameters, dtypes)
    tmp_12_df['value'] = 1.0 - (
        1.0 - tmp_12_df[f'infectious_proba_{time_suffix}']
    ) ** tmp_12_df[f'num_people_seen_{suffix}']

    # Add the complementary event (at_least_one_inf = 0)
    tmp_12_df[new_key_12] = 1.0
    copy = tmp_12_df.copy()
    copy[new_key_12] = 0
    copy['value'] = 1.0 - copy['value']

    tmp_12_df = pd.concat([
        tmp_12_df, copy
    ])

    keys = [
        new_key_9,
        new_key_10,
        new_key_11,
        new_key_12,
    ]

    dfs = [
        tmp_9_df,
        tmp_10_df,
        tmp_11_df,
        tmp_12_df,
    ]

    add_dfs_to_bn(
        bayesian_network,
        dfs,
        keys,
        storage_folder=storage_folder
    )

    return {
        'at_least_one_inf_key': new_key_12,
        'at_least_one_inf_df': tmp_12_df
    }


def create_longitudinal(
    dates,
    person,
    bayesian_network,
    storage_folder=None
):
    """
    Parameters:
        dates: pd.Series([pd.Date])
        person: string
        bayesian_network: BayesianNetwork
    """
    # TODO: the dose_key might depend on how many people there are in a
    # household.
    event = 'work'
    time_format = '%m-%d-%y'

    person_index = index_name(person)
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

    start_symp_key = f'start_symp_{person_index}'
    end_symp_key = f'end_symp_{person_index}'

    start_symp_df = create_start_symp(
        immunity_key=immunity_key,
        start_symp_key=start_symp_key

    )

    end_symp_df = create_end_symp(
        immunity_key=immunity_key,
        end_symp_key=end_symp_key
    )

    keys = [
        viral_load_n_key,
        viral_load_p_key,
        immunity_factor_key,
        start_symp_key,
        end_symp_key
    ]

    dfs = [
        viral_load_n_df,
        viral_load_p_df,
        immunity_factor_df,
        start_symp_df,
        end_symp_df
    ]

    add_dfs_to_bn(
        bayesian_network,
        dfs,
        keys,
        storage_folder=storage_folder
    )

    for date in dates:
        date_event_others_to_person = index_name(
            date.strftime(time_format),
            event,
            'others',
            person
        )
        dose_key = f"dose_tmp_13_{date_event_others_to_person}"

        create_dose_from_strangers(
            time=date,
            person=person,
            event=event,
            bayesian_network=bayesian_network,
            storage_folder=storage_folder,
            time_format=time_format
        )

        # TODO: create interactions between known individuals.
        # Then add the sum of doses from individuals with dose from strangers.
        # We could have this part be delegated to a function to make it more
        # flexible.

        # Scenario 1: The person is a loner and doesn't have friends. In that
        # point of view, everyone else is a stranger.

        # Scenario 2: The person is not a loner. The person hangs out with only
        # known people.

        # Scenario 3: The person hangs out with a mix of known people and
        # strangers (e.g. at a restaurant).
        #     Person and known people share
        #         - location/gathering type
        #         - ventilation
        #         - volume
        #         - activity
        #         - number of people seen
        #         - unreported cases
        #         - duration hyperprior?


        create_inf_dsi_viral_load_measurements(
            person=person,
            time=date,
            dose_key=dose_key,
            bayesian_network=bayesian_network,
            time_format=time_format,
            viral_load_n_key=viral_load_n_key,
            viral_load_p_key=viral_load_p_key,
            viral_load_n_df=viral_load_n_df,
            viral_load_p_df=viral_load_p_df,
            immunity_key=immunity_key,
            immunity_factor_key=immunity_factor_key,
            immunity_factor_df=immunity_factor_df,
            start_symp_key=start_symp_key,
            end_symp_key=end_symp_key,
            start_symp_df=start_symp_df,
            end_symp_df=end_symp_df,
        )


def create_event_mapping(
    outcome_name,
    outcome_mappings,
    event_suffix
):
    """
    Useful for dynamically generating CPTs related to the event.
    Some CPTs are stable -- anyone could use them. However, some CPTs are very
    specific to other people, so it makes sense to sometimes not have
    hard-coded CPTs
    """
    collection = []

    for event, obj in outcome_mappings.items():
        for outcome, proba in obj.items():
            collection.append({
                f'event_{event_suffix}': event,
                outcome_name: outcome,
                'value': proba
            })

    return pd.DataFrame(collection)


def create_duration_between_two_people(
    time,
    event,
    person_1,
    person_2,
    duration_unique
):
    key_1 = f'duration_{index_name(time, event, person_1)}'
    key_2 = f'duration_{index_name(time, event, person_2)}'
    min_dur_key = f'min_duration_{index_name(time, event, person_1, person_2)}'

    parameters = {
        key_1: duration_unique,
        key_2: duration_unique
    }

    dtypes = {
        key_1: 'float64',
        key_2: 'float64',
    }

    df = mega_join_cross_product(parameters, dtypes)
    df[min_dur_key] = \
        df.apply(lambda x: min(x[key_1], x[key_2]), axis=1)

    return min_dur_key, df


def create_lim_vol_vent(
    time,
    event,
    vol_vent_mult_unique,
    vol_vent_mult_key,
    dist_unique,
    people
):
    """
    Limit the effect of volume and ventilation depending on the distance
    between two people. If distance is greater than 2 meters (6 ft), then
    volume and ventilation have full effect. Otherwise, the effect gets
    dampened. The closer the person to each other, the less volume and
    ventilation have an effect in reducing the dose that one person gets from
    another.
    """
    person_1 = people[0]
    person_2 = people[1]

    distance_key = f'distance_{index_name(time, event, person_1, person_2)}'
    lim_mult_key = f'lim_mult_vv_{index_name(time, event, person_1, person_2)}'

    parameters = {
        distance_key: dist_unique,
        vol_vent_mult_key: vol_vent_mult_unique
    }

    dtypes = {
        distance_key: 'float64',
        vol_vent_mult_key: 'float64',
    }

    df = mega_join_cross_product(parameters, dtypes)

    tmp_key = 'tmp_key'

    df[tmp_key] = 1.0

    # check how many meters away from each other

    # TODO: could refactor using pd.cut here
    df[tmp_key].mask(df[distance_key] >= 2.0, 6.0 / 6)

    for i in range(5, -1, -1):
        df[tmp_key].mask(
            (df[distance_key] > 2.0 * i / 6.0) \
            & (df[distance_key] <= 2.0 * (i + 1) / 6.0),
            i / 6.0
        )

    df[lim_mult_key] = df[vol_vent_mult_key] * df[tmp_key]

    return lim_mult_key, df[[vol_vent_mult_key, lim_mult_key, 'value']]


def create_dose_from_people(
    time,
    friends,
    event,
    bayesian_network,
    storage_folder=None,
    time_format='%m-%d-%y',
    rounding=5
):
    time_str = time.strftime(time_format)

    time_event_index = index_name(time_str, event)
    volume_key = f'volume_{time_event_index}'
    ventilation_key = f'ventilation_{time_event_index}'
    vol_vent_mult_key = f'vol_vent_mult_{time_event_index}'
    vol_vent_key = f'vol_vent_{time_event_index}'

    mappings = {
        'volume': {
            'mighty squirrel': {
                10_000: 1.0
            },
            "ruberic's living room": {
                30: 1.0
            }
        },
        'ventilation': {
            'mighty squirrel': {
                0.5: 0.5,
                1.0: 0.5,
            },
            "ruberic's living room": {
                30: 1.0
            }
        }
    }

    key_to_df = {}

    volume_df = create_event_mapping(
        outcome_name=volume_key,
        outcome_mappings=mappings['volume'],
        event_suffix=time_event_index
    )

    key_to_df[volume_key] = volume_df

    ventilation_df = create_event_mapping(
        outcome_name=ventilation_key,
        outcome_mappings=mappings['ventilation'],
        event_suffix=time_event_index
    )

    key_to_df[ventilation_key] = ventilation_df

    multiplied_df = multiply_by(
        factor_1_unique=volume_df[volume_key].unique(),
        factor_1_name=volume_key,
        factor_2_unique=ventilation_df[ventilation_key].unique(),
        factor_2_name=ventilation_key,
        new_key=vol_vent_mult_key
    )

    key_to_df[vol_vent_mult_key] = multiplied_df

    # Add branch for others.
    # Loop through friends.Connect them
    quanta_unique = list(range(0, 90, 10))
    duration_unique = [
        0.25, 0.5, 1, 2, 3, 5, 8, 13, 21
    ]

    dist_unique = np.arange(0.0, 2.0, 1.0 / 6.0).round(2)

    mask_unique = [1, 0.7, 0.4, 0.10, 0.01]
    giver_dict = {}

    for giver in friends:
        time_giver_index = index_name(time_str, giver)
        time_event_giver_index = index_name(time_str, event, giver)

        # Giver
        quanta_key = f'quanta_{time_giver_index}'

        res = create_activity_exhalation(
            suffix=time_event_giver_index,
            bayesian_network=bayesian_network,
            storage_folder=storage_folder
        )

        tmp_0_key = f"tmp_0_{time_event_giver_index}"
        exhalation_df = multiply_by(
            factor_1_unique=quanta_unique,
            factor_1_name=quanta_key,
            factor_2_unique=res['activity_exhalation_df'][
                res['exhalation_factor_key']
            ].unique(),
            factor_2_name=res['exhalation_factor_key'],
            new_key=tmp_0_key
        )

        key_to_df[tmp_0_key] = exhalation_df

        tmp_1_key = f'tmp_1_{time_event_giver_index}'
        mask_key = f'mask_{time_event_giver_index}'

        tmp_1_df = multiply_by(
            factor_1_unique=exhalation_df[tmp_0_key].unique(),
            factor_1_name=tmp_0_key,
            factor_2_unique=mask_unique,
            factor_2_name=mask_key,
            new_key=tmp_1_key
        )

        giver_dict[giver] = {
            'key': tmp_0_key,
            'df': exhalation_df
        }

        key_to_df[tmp_1_key] = tmp_1_df

    visited_pair = {}

    for giver in friends:
        for receiver in friends:
            if giver == receiver:
                continue

            sorted_key = '-'.join(sorted([giver, receiver]))

            if sorted_key not in visited_pair:
                min_dur_key, min_dur_df = create_duration_between_two_people(
                    time=time_str,
                    event=event,
                    person_1=giver,
                    person_2=receiver,
                    duration_unique=duration_unique
                )

                key_to_df[min_dur_key] = min_dur_df

                visited_pair[sorted_key] = {
                    'key': min_dur_key,
                    'df': min_dur_df
                }

                vol_vent_mult_df = key_to_df[vol_vent_mult_key]
                lim_mult_key, lim_df = create_lim_vol_vent(
                    time_str,
                    event,
                    vol_vent_mult_unique=vol_vent_mult_df[vol_vent_mult_key],
                    vol_vent_mult_key=vol_vent_mult_key,
                    dist_unique=dist_unique,
                    people=[giver, receiver]
                )

                key_to_df[lim_mult_key] = lim_df

    for giver in friends:
        time_event_giver_index = index_name(time_str, event, giver)

        for receiver in friends:
            if giver == receiver:
                continue

            time_event_receiver_index = index_name(
                time_str,
                event,
                receiver
            )

            time_event_giver_to_receiver = index_name(
                time_str,
                event,
                giver,
                receiver
            )


            sorted_names = sorted([giver, receiver])
            sorted_key = '-'.join(sorted_names)

            lim_mult_index = index_name(
                time_str,
                event,
                sorted_names[0],
                sorted_names[1]
            )

            lim_mult_key = f'lim_mult_vv_{lim_mult_index}'

            mask_key = f'mask_{time_event_receiver_index}'
            # Add duration
            duration_pair = visited_pair[sorted_key]
            duration_pair_key = duration_pair['key']
            duration_pair_df = duration_pair['df']

            tmp_1_key = f'tmp_1_{time_event_giver_index}'

            tmp_5_key = f'tmp_5_{time_event_giver_to_receiver}'
            tmp_5_df = multiply_by(
                factor_1_unique=duration_pair_df[duration_pair_key].unique(),
                factor_1_name=duration_pair_key,
                factor_2_unique=key_to_df[tmp_1_key][tmp_1_key].unique(),
                factor_2_name=tmp_1_key,
                new_key=tmp_5_key
            )
            # TODO: round to preserve memory
            key_to_df[tmp_5_key] = tmp_5_df
            round_column(tmp_5_df, tmp_5_key, rounding=5)

            # Add inhalation
            inhalation_rate_key = \
                f'inhalation_rate_{time_event_receiver_index}'

            activity_specific_df = create_activity_specific_breathing_rate_df(
                receiver,
                time_str,
                event,
                inhalation_rate_key
            )
            key_to_df[inhalation_rate_key] = activity_specific_df

            tmp_6_key = f'tmp_6_{time_event_giver_to_receiver}'
            tmp_6_df = multiply_by(
                factor_1_unique=tmp_5_df[tmp_5_key].unique(),
                factor_1_name=tmp_5_key,
                factor_2_unique=activity_specific_df[inhalation_rate_key].unique(),
                factor_2_name=inhalation_rate_key,
                new_key=tmp_6_key
            )
            round_column(tmp_6_df, tmp_6_key, rounding=5)

            key_to_df[tmp_6_key] = tmp_6_df
            # Add effect of masking the inhaler
            tmp_7_key = f'tmp_7_{time_event_giver_to_receiver}'
            tmp_7_df = multiply_by(
                factor_1_unique=mask_unique,
                factor_1_name=mask_key,
                factor_2_unique=tmp_6_df[tmp_6_key].unique(),
                factor_2_name=tmp_6_key,
                new_key=tmp_7_key
            )
            round_column(tmp_7_df, tmp_7_key, rounding=5)
            key_to_df[tmp_7_key] = tmp_7_df
            # TODO: pair limited ventilation / volume with

            giver_data = giver_dict[giver]

            tmp_8_key = f'tmp_8_{time_event_giver_to_receiver}'
            tmp_8_df = divide_by(
                factor_1_unique=tmp_7_df[tmp_7_key].unique(),
                factor_1_name=tmp_7_key,
                factor_2_unique=key_to_df[lim_mult_key][lim_mult_key].unique(),
                factor_2_name=lim_mult_key,
                new_key=tmp_8_key
            )

            round_column(tmp_8_df, tmp_8_key, rounding=5)
            key_to_df[tmp_8_key] = tmp_8_df

    add_dfs_to_bn(
        bayesian_network=bayesian_network,
        dfs=list(key_to_df.values()),
        keys=list(key_to_df.keys()),
        storage_folder=storage_folder
    )

    # sum_up_doses_of_people(
        # friends,
        # time_str,
        # event,
        # bayesian_network,
        # prefix=f'tmp_8',
        # rounding=4,
        # storage_folder=storage_folder
    # )
#
    # TODO: sum the doses.


def create_dose_from_strangers(
    time,
    person,
    event,
    bayesian_network,
    time_format,
    storage_folder=None,
    rounding=5
):
    time_str = time.strftime(time_format)

    volume_vent_key, volume_vent_event_df = create_room_event(
        event_suffix=index_name(time_str, person, event),
        bayesian_network=bayesian_network
    )

    suffix = index_name(time_str, event, "others", person)
    date_event_self_suffix = index_name(time_str, event, person)

    res = create_activity_exhalation(
        suffix,
        bayesian_network,
        storage_folder=storage_folder
    )
    prefix = 'dose_tmp'

    new_key_1 = f"{prefix}_1_{suffix}"
    tmp_1_df = multiply_by(
        factor_1_unique=volume_vent_event_df[volume_vent_key].unique(),
        factor_1_name=volume_vent_key,
        factor_2_unique=res['activity_exhalation_df'][
            res['exhalation_factor_key']
        ].unique(),
        factor_2_name=res['exhalation_factor_key'],
        new_key=new_key_1
    )
    round_column(tmp_1_df, new_key_1, rounding=rounding)

    inhalation_rate_key = \
        f'inhalation_rate_{index_name(time_str, event, person)}'

    tmp_2_df = create_activity_specific_breathing_rate_df(
        person,
        time_str,
        event,
        inhalation_rate_key
    )

    new_key_3 = f"{prefix}_3_{suffix}"
    tmp_3_df = multiply_by(
        factor_1_unique=tmp_2_df[inhalation_rate_key].unique(),
        factor_1_name=inhalation_rate_key,
        factor_2_unique=tmp_1_df[new_key_1].unique(),
        factor_2_name=new_key_1,
        new_key=new_key_3
    )

    round_column(tmp_3_df, new_key_3, rounding=4)

    new_key_4 = f'{prefix}_4_{suffix}'
    tmp_4_df = multiply_by(
        factor_1_unique=tmp_3_df[new_key_3].unique(),
        factor_1_name=new_key_3,
        factor_2_unique=[
            0.25, 0.5, 1, 2, 3, 5, 8, 13, 21
        ],
        factor_2_name=f'duration_{suffix}',
        new_key=new_key_4
    )
    round_column(tmp_4_df, new_key_4, rounding=4)

    new_key_5 = f'{prefix}_5_{suffix}'
    tmp_5_df = multiply_by(
        factor_1_unique=tmp_4_df[new_key_4].unique(),
        factor_1_name=new_key_4,
        factor_2_unique=[
            0, 10, 20, 30, 40, 50, 60, 70, 80
        ],
        factor_2_name=f'quanta_{suffix}',
        new_key=new_key_5
    )
    round_column(tmp_5_df, new_key_5, rounding=4)

    tmp_6_df, new_key_6 = create_avg_mask_reduction_from_others(suffix)

    new_key_7 = f'{prefix}_7_{suffix}'
    tmp_7_df = multiply_by(
        factor_1_unique=tmp_5_df[new_key_5].unique(),
        factor_1_name=new_key_5,
        factor_2_unique=tmp_6_df[new_key_6].unique(),
        factor_2_name=new_key_6,
        new_key=new_key_7
    )
    round_column(tmp_7_df, new_key_7, rounding=4)
    cap(df=tmp_7_df, key=new_key_7, maximum=1000)

    new_key_8 = f'{prefix}_8_{suffix}'
    tmp_8_df = multiply_by(
        factor_1_unique=tmp_7_df[new_key_7].unique(),
        factor_1_name=new_key_7,
        factor_2_unique=[1, 0.7, 0.4, 0.10, 0.01],
        factor_2_name=f'mask_{date_event_self_suffix}',
        new_key=new_key_8
    )
    round_column(tmp_8_df, new_key_8, rounding=4)
    cap(df=tmp_8_df, key=new_key_8, maximum=1000)

    dictionary = create_at_least_one_inf(
        time_str,
        prefix,
        suffix,
        bayesian_network,
        storage_folder=storage_folder
    )

    new_key_12 = dictionary['at_least_one_inf_key']
    tmp_12_df = dictionary['at_least_one_inf_df']

    new_key_13 = f'{prefix}_13_{suffix}'
    tmp_13_df = multiply_by(
        factor_1_unique=tmp_12_df[new_key_12].unique(),
        factor_1_name=new_key_12,
        factor_2_unique=tmp_8_df[new_key_8].unique(),
        factor_2_name=new_key_8,
        new_key=new_key_13
    )

    keys = [
        new_key_1,
        inhalation_rate_key,
        new_key_3,
        new_key_4,
        new_key_5,
        new_key_6,
        new_key_7,
        new_key_8,
        new_key_13
    ]

    dfs = [
        tmp_1_df,
        tmp_2_df,
        tmp_3_df,
        tmp_4_df,
        tmp_5_df,
        tmp_6_df,
        tmp_7_df,
        tmp_8_df,
        tmp_13_df
    ]

    add_dfs_to_bn(
        bayesian_network,
        dfs,
        keys,
        storage_folder=storage_folder
    )

    return new_key_13


def create_infection_from_dose(suffix, dose_key, infected_key):
    """
    Compute probability of infection given a dose.

    Parameters:
        suffix: string

    Returns: pd.DataFrame
    """
    infected_key = f"infected_{suffix}"

    parameters = {
        dose_key: np.arange(0.0, 10.0, 0.0001).round(4),
        infected_key: [0.0, 1.0],
    }

    dtypes = {
        dose_key: 'float64',
        infected_key: 'int8',
    }

    df = mega_join_cross_product(
        parameters,
        dtypes
    )

    df['value'] = df['value'].mask(
        df[infected_key] == 1,
        1 - np.exp(-df[dose_key])
    )
    df['value'] = df['value'].mask(
        df[infected_key] == 0,
        np.exp(-df[dose_key])
    )

    return df


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
                0, 10, 20, 30, 40, 50, 60, 70, 80
            ],
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
    suffix,
    bayesian_network,
    storage_folder=None
):
    exhalation_factor_key = \
        f'exhalation_factor_{suffix}'

    activity_exhalation_df = pd.DataFrame({
        f'activity_exhalation_{suffix}': [
            "Resting - Oral breathing",
            "Resting - Speaking",
            "Resting - Loudly speaking",
            "Standing - Oral breathing",
            "Standing - Speaking",
            "Standing - Loudly speaking",
            "Light exercise - Oral breathing",
            "Light exercise - Speaking",
            "Light exercise - Loudly speaking",
            "Heavy exercise - Oral breathing",
            "Heavy exercise - Speaking",
            "Heavy exercise - Loudly speaking"
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
        'exhalation_factor_key': exhalation_factor_key,
        'activity_exhalation_df': activity_exhalation_df
    }


def create_doses(
    people,
    time,
    event,
    volume_vent_key,
    volume_vent_df,
    bayesian_network,
    time_format,
    storage_folder=None,
    prefix='dose_tmp',
):

    time_str = time.strftime(time_format)

    for p1 in people:
        suffix = index_name(time, event, p1)
        activity_exh_dict = create_activity_exhalation(
            suffix,
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
                time_str,
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

    # TODO: return the key of the dose so that it can be connected to the
    # infected key.
    sum_up_doses_of_people(
        people,
        time_str,
        event,
        bayesian_network,
        prefix=f'{prefix}_9',
        rounding=4,
        storage_folder=storage_folder
    )


def sum_up_doses_of_people(
    people,
    time,
    event,
    bayesian_network,
    prefix="tmp_9",
    rounding=5,
    max_val=10,
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
                key_1: np.arange(0.0, float(max_val), 1.0 / rounding).round(rounding),
                key_2: np.arange(0.0, float(max_val), 1.0 / rounding).round(rounding),
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

            cap(df=df, key=new_key, maximum=max_val)
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


def create_bayesian_network(events_with_priors):
    bn = BN()
    person_priors = events_with_priors['person_priors']

    gatherings = events_with_priors['gatherings']
    for date, obj_1 in gatherings.items():
        import pdb; pdb.set_trace()


def create_event_mixed(
    friends,
):
    """
    Add nodes and edges to Bayesian network such that people who are in the
    system (friends) can affect each other's doses, but also have other
    individuals affect the listed friends.
    """
    pass


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


if __name__ == '__main__':
    bayesian_network = BN(graphviz_dag=graphviz.Digraph())
