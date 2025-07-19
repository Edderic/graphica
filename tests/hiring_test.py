import logging

import graphviz
import pandas as pd

from ..graphica.data import ParquetData
from ..graphica.ds import BayesianNetwork, \
    ConditionalProbabilityTable as CPT, Query
from ..graphica.inference import VariableElimination
from .conftest import (assert_approx_value_df, clean_tmp, create_df_medium,
                       create_prior_df, get_tmp_path)


def create_levels_df_mini():
    collection = []
    for narrative in range(5):
        for viz in range(5):
            for level in range(5):
                true_level = min(
                    narrative,
                    viz,
                )
                if level == true_level:
                    value = 1.0

                    collection.append({
                        'Narrative': narrative,
                        'Visualization': viz,
                        'Level': level,
                        'value': value
                    })

    return pd.DataFrame(collection)


def create_levels_df_simple():
    collection = []

    for narrative in range(5):
        for gen_eng_skills in range(5):
            for level in range(5):
                minimum = min(
                    narrative,
                    gen_eng_skills
                )

                if level == minimum:
                    value = 1

                    collection.append({
                        'Narrative': narrative,
                        'General Engineering Skills':
                            gen_eng_skills,
                        'Level': level,
                        'value': value
                    })

    return pd.DataFrame(collection)


def test_hiring_sub_2():
    cpts = {}

    prior_keys = [
        'Narrative',
        'Visualization',
    ]

    for prior in prior_keys:
        cpts[prior] = CPT(
            table=create_prior_df(outcome=prior),
            outcomes=[prior],
        )

    med_cpts = [
        ('Narrative', 'Narrative Score 1'),
        ('Narrative', 'Narrative Score 2'),
        ('Visualization', 'Visualization Score 1'),
        ('Visualization', 'Visualization Score 2'),
    ]

    for start_var, end_var in med_cpts:
        cpts[end_var] = CPT(
            ParquetData(
                create_df_medium(given=start_var, outcome=end_var),
                storage_folder=get_tmp_path()
            ),
            outcomes=[end_var],
            givens=[start_var]
        )

    levels_df = create_levels_df_mini()

    cpts['Level'] = CPT(
        ParquetData(levels_df, storage_folder=get_tmp_path()),
        outcomes=['Level'],
        givens=[
            'Narrative',
            'Visualization',
        ]
    )

    bayesian_network = BayesianNetwork(
        random_variables=cpts
    )

    logging.basicConfig(level=logging.DEBUG)

    query_2 = Query(
        givens=['Narrative'],
        outcomes=['Level']
    )

    result_2 = VariableElimination(
        network=bayesian_network,
        query=query_2
    ).compute()

    expected_df_2 = pd.DataFrame([
        {'Level': 0, 'value': 1.0, 'Narrative': 0},
        {'Level': 0, 'value': 0.2, 'Narrative': 1},
        {'Level': 1, 'value': 0.8, 'Narrative': 1},
        {'Level': 0, 'value': 0.2, 'Narrative': 2},
        {'Level': 1, 'value': 0.2, 'Narrative': 2},
        {'Level': 2, 'value': 0.6, 'Narrative': 2},
        {'Level': 0, 'value': 0.2, 'Narrative': 3},
        {'Level': 1, 'value': 0.2, 'Narrative': 3},
        {'Level': 2, 'value': 0.2, 'Narrative': 3},
        {'Level': 3, 'value': 0.4, 'Narrative': 3},
        {'Level': 0, 'value': 0.2, 'Narrative': 4},
        {'Level': 1, 'value': 0.2, 'Narrative': 4},
        {'Level': 2, 'value': 0.2, 'Narrative': 4},
        {'Level': 3, 'value': 0.2, 'Narrative': 4},
        {'Level': 4, 'value': 0.2, 'Narrative': 4},
    ])

    assert_approx_value_df(
        actual_df=result_2.get_df(),
        expected_df=expected_df_2
    )

    clean_tmp()


def test_hiring_sub():
    cpts = {}

    priors = [
        'Narrative',
        'Visualization',
    ]

    priors = {}
    for prior in priors:
        priors[prior] = CPT(
            ParquetData(
                create_prior_df(outcome=prior),
                storage_folder=get_tmp_path()
            ),
            outcomes=[prior],
        )

    med_cpts = [
        ('Narrative', 'Narrative Score 1'),
        ('Narrative', 'Narrative Score 2'),
        ('Visualization', 'Visualization Score 1'),
        ('Visualization', 'Visualization Score 2'),
    ]

    for start_var, end_var in med_cpts:
        cpts[end_var] = CPT(
            ParquetData(
                create_df_medium(given=start_var, outcome=end_var),
                storage_folder=get_tmp_path()
            ),
            outcomes=[end_var],
            givens=[start_var]
        )

    levels_df = create_levels_df_mini()

    cpts['Level'] = CPT(
        ParquetData(
            levels_df,
            storage_folder=get_tmp_path()
        ),
        outcomes=['Level'],
        givens=[
            'Narrative',
            'Visualization',
        ]
    )

    bayesian_network = BayesianNetwork()
    bayesian_network.add_nodes(cpts)
    bayesian_network.add_nodes(priors)

    logging.basicConfig(level=logging.DEBUG)

    query_1 = Query(
        givens=[{
            'Narrative': lambda df: df['Narrative'] == 4
        }],
        outcomes=['Level']
    )

    variable_elimination = VariableElimination(
        network=bayesian_network,
        query=query_1
    )
    result_1 = variable_elimination.compute()
    # Because
    expected_result = pd.DataFrame([
        {'Level': 0, 'Narrative': 4, 'value': 0.2},
        {'Level': 1, 'Narrative': 4, 'value': 0.2},
        {'Level': 2, 'Narrative': 4, 'value': 0.2},
        {'Level': 3, 'Narrative': 4, 'value': 0.2},
        {'Level': 4, 'Narrative': 4, 'value': 0.2},
    ])

    assert_approx_value_df(
        actual_df=result_1.get_df(),
        expected_df=expected_result
    )

    query_2 = Query(
        givens=['Narrative'],
        outcomes=['Level']
    )

    result_2 = VariableElimination(
        network=bayesian_network,
        query=query_2
    ).compute()

    expected_df_2 = pd.DataFrame([
        {'Level': 0, 'value': 1.0, 'Narrative': 0},
        {'Level': 0, 'value': 0.2, 'Narrative': 1},
        {'Level': 1, 'value': 0.8, 'Narrative': 1},
        {'Level': 0, 'value': 0.2, 'Narrative': 2},
        {'Level': 1, 'value': 0.2, 'Narrative': 2},
        {'Level': 2, 'value': 0.6, 'Narrative': 2},
        {'Level': 0, 'value': 0.2, 'Narrative': 3},
        {'Level': 1, 'value': 0.2, 'Narrative': 3},
        {'Level': 2, 'value': 0.2, 'Narrative': 3},
        {'Level': 3, 'value': 0.4, 'Narrative': 3},
        {'Level': 0, 'value': 0.2, 'Narrative': 4},
        {'Level': 1, 'value': 0.2, 'Narrative': 4},
        {'Level': 2, 'value': 0.2, 'Narrative': 4},
        {'Level': 3, 'value': 0.2, 'Narrative': 4},
        {'Level': 4, 'value': 0.2, 'Narrative': 4},
    ])

    assert_approx_value_df(
        actual_df=result_2.get_df(),
        expected_df=expected_df_2
    )

    result_2.get_df().merge(expected_df_2, on=['Narrative', 'Level'])

    query_3 = Query(
        givens=[
            {
                'Narrative Score 1':
                lambda df: df['Narrative Score 1'] == 4
            },
            {
                'Visualization Score 1':
                lambda df: df['Visualization Score 1'] == 4
            }
        ],
        outcomes=['Level']
    )

    result_3 = VariableElimination(
        network=bayesian_network,
        query=query_3
    ).compute()

    expected_df_3 = pd.DataFrame([
        {
            'Visualization Score 1': 4,
            'value': 0.102493,
            'Narrative Score 1': 4,
            'Level': 3
        },
        {
            'Visualization Score 1': 4,
            'value': 0.897507,
            'Narrative Score 1': 4,
            'Level': 4
        }
    ])

    assert_approx_value_df(
        actual_df=result_3.get_df(),
        expected_df=expected_df_3
    )


def test_hiring_simple():
    medium_cpt_pairs = [
        ('Narrative', 'Narrative Score 1'),
        ('General Engineering Skills', 'Reproducible Programming'),
        ('Reproducible Programming', "Git commits are semantic"),
        ('Git commits are semantic', 'Git commits are semantic Score 1')
    ]

    cpts = []

    for start, end in medium_cpt_pairs:
        df = create_df_medium(start, end)
        cpts.append(
            CPT(
                ParquetData(
                    df,
                    storage_folder=get_tmp_path()
                ),
                givens=[start],
                outcomes=[end]
            )
        )

    roots = ['Narrative', 'General Engineering Skills']

    for outcome in roots:
        cpts.append(
            CPT(
                ParquetData(
                    create_prior_df(outcome=outcome),
                    storage_folder=get_tmp_path()
                ),
                outcomes=[outcome]
            )
        )

    cpts.append(
        CPT(
            ParquetData(
                create_levels_df_simple(),
                storage_folder=get_tmp_path()
            ),
            outcomes=['Level'],
            givens=['General Engineering Skills', 'Narrative']
        )
    )

    bayesian_network = BayesianNetwork()
    bayesian_network.add_nodes(cpts)

    result = VariableElimination(
        network=bayesian_network,
        query=Query(
            outcomes=['Level'],
            givens=[
                {'Narrative Score 1': 4},
                {'Git commits are semantic Score 1': 4}
            ]
        )
    ).compute()

    result_df = result.get_df()
    indexed = result_df.set_index(
        ['Narrative Score 1', 'Git commits are semantic Score 1', 'Level']
    )

    assert indexed.xs((4, 4, 1))['value'] < indexed.xs((4, 4, 2))['value']
    assert indexed.xs((4, 4, 2))['value'] < indexed.xs((4, 4, 3))['value']
    assert indexed.xs((4, 4, 3))['value'] < indexed.xs((4, 4, 4))['value']
