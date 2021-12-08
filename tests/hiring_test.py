import logging

import pandas as pd

from ..linx.ds import BayesianNetwork, \
    ConditionalProbabilityTable as CPT, Query
from ..linx.infer import VariableElimination
from .conftest import create_df_easy, create_df_medium, create_df_hard, \
    create_prior_df, assert_approx_value_df


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
                else:
                    value = 0.0

                collection.append({
                    'Narrative': narrative,
                    'Visualization': viz,
                    'Level': level,
                    'value': value
                })

    return pd.DataFrame(collection)


def create_levels_df():
    collection = []

    # TODO:
    # Some skills are easier to get, and some skills aren't necessarily needed
    # for a high level
    for narrative in range(5):
        for viz in range(5):
            for stats in range(5):
                for data_management in range(5):
                    for equity in range(5):
                        for gen_eng_skills in range(5):
                            for learn in range(5):
                                for level in range(5):
                                    minimum = min(
                                        narrative,
                                        viz,
                                        stats,
                                        data_management,
                                        equity,
                                        gen_eng_skills,
                                        learn
                                    )

                                    if minimum == 0:
                                        value = 0

                                    else:
                                        summation = sum(
                                            [
                                                narrative,
                                                viz,
                                                stats,
                                                data_management,
                                                equity,
                                                gen_eng_skills,
                                                learn
                                            ]
                                        )

                                        mean = summation / 7.0
                                        proposed_level = round(mean)
                                        value = (proposed_level == level) * 1.0

                                    collection.append({
                                        'Narrative': narrative,
                                        'Visualization': viz,
                                        'Statistics': stats,
                                        'Data Management': data_management,
                                        'Equity': equity,
                                        'General Engineering Skills':
                                            gen_eng_skills,
                                        'Learn': learn,
                                        'Level': level,
                                        'value': value
                                    })

    return pd.DataFrame(collection)


def test_hiring_sub():
    cpts = {}

    priors = [
        'Narrative',
        'Visualization',
    ]

    priors = {}
    for prior in priors:
        priors[prior] = CPT(
            df=create_prior_df(outcome=prior),
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
            df=create_df_medium(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    levels_df = create_levels_df_mini()

    cpts['Level'] = CPT(
        df=levels_df,
        outcomes=['Level'],
        givens=[
            'Narrative',
            'Visualization',
        ]
    )

    bayesian_network = BayesianNetwork(
        cpts=list(cpts.values()),
        priors=list(priors.values())
    )

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
        actual_df=result_1.df,
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
        {'Level': 1, 'value': 0.0, 'Narrative': 0},
        {'Level': 2, 'value': 0.0, 'Narrative': 0},
        {'Level': 3, 'value': 0.0, 'Narrative': 0},
        {'Level': 4, 'value': 0.0, 'Narrative': 0},
        {'Level': 0, 'value': 0.2, 'Narrative': 1},
        {'Level': 1, 'value': 0.8, 'Narrative': 1},
        {'Level': 2, 'value': 0.0, 'Narrative': 1},
        {'Level': 3, 'value': 0.0, 'Narrative': 1},
        {'Level': 4, 'value': 0.0, 'Narrative': 1},
        {'Level': 0, 'value': 0.2, 'Narrative': 2},
        {'Level': 1, 'value': 0.2, 'Narrative': 2},
        {'Level': 2, 'value': 0.6, 'Narrative': 2},
        {'Level': 3, 'value': 0.0, 'Narrative': 2},
        {'Level': 4, 'value': 0.0, 'Narrative': 2},
        {'Level': 0, 'value': 0.2, 'Narrative': 3},
        {'Level': 1, 'value': 0.2, 'Narrative': 3},
        {'Level': 2, 'value': 0.2, 'Narrative': 3},
        {'Level': 3, 'value': 0.4, 'Narrative': 3},
        {'Level': 4, 'value': 0.0, 'Narrative': 3},
        {'Level': 0, 'value': 0.2, 'Narrative': 4},
        {'Level': 1, 'value': 0.2, 'Narrative': 4},
        {'Level': 2, 'value': 0.2, 'Narrative': 4},
        {'Level': 3, 'value': 0.2, 'Narrative': 4},
        {'Level': 4, 'value': 0.2, 'Narrative': 4},
    ])

    assert_approx_value_df(
        actual_df=result_2.df,
        expected_df=expected_df_2
    )

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
            'value': 0.000000,
            'Narrative Score 1': 4,
            'Level': 0
        },
        {
            'Visualization Score 1': 4,
            'value': 0.000000,
            'Narrative Score 1': 4,
            'Level': 1
        },
        {
            'Visualization Score 1': 4,
            'value': 0.000000,
            'Narrative Score 1': 4,
            'Level': 2
        },
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
        actual_df=result_3.df,
        expected_df=expected_df_3
    )


def test_hiring():
    cpts = {}

    priors = [
        'Narrative',
        'Visualization',
        'Statistics',
        'Data Management',
        'General Engineering Skills',
        'Equity',
        'Learn'
    ]

    priors = {}
    for prior in priors:
        priors[prior] = CPT(
            df=create_prior_df(outcome=prior),
            outcomes=[prior],
        )

    easy_cpts = [
        ('Data Management', 'General knowledge of DB & Cloud arch.'),
        ('Reproducible Programming', 'Use conditional logic'),
        ('Reproducible Programming', 'Use of functions'),
        ('Programming Readability', 'Code functionality'),
    ]

    for start_var, end_var in easy_cpts:
        cpts[end_var] = CPT(
            df=create_df_easy(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    med_cpts = [
        ('Programming Readability', 'Idiomatic Code'),
        ('Idiomatic Code', 'Idiomatic Code Score 1'),
        ('Idiomatic Code', 'Idiomatic Code Score 2'),
        ('Programming Readability', 'Code efficiency'),
        ('Code efficiency', 'Code efficiency Score 1'),
        ('Code efficiency', 'Code efficiency Score 2'),
        ('Programming Readability', 'OOP'),
        ('OOP', 'OOP Score 1'),
        ('OOP', 'OOP Score 2'),
        ('Programming Readability', 'DRY'),
        ('DRY', 'DRY Score 1'),
        ('DRY', 'DRY Score 2'),
        ('Code functionality', 'Code functionality Score 1'),
        ('Code functionality', 'Code functionality Score 2'),
        ('Narrative', 'Narrative Score 1'),
        ('Narrative', 'Narrative Score 2'),
        ('Visualization', 'Visualization Score 1'),
        ('Visualization', 'Visualization Score 2'),
        ('Statistics', 'Statistics Score 1'),
        ('Statistics', 'Statistics Score 2'),
        ('Reproducible Programming', 'Git commits are semantic'),
        ('Git commits are semantic', 'Git commits are semantic Score 1'),
        ('Git commits are semantic', 'Git commits are semantic Score 2'),
        ('Reproducible Programming', 'Uses a package manager'),
        ('Uses a package manager', 'Uses a package manager Score 1'),
        ('Uses a package manager', 'Uses a package manager Score 2'),
        ('Reproducible Programming', 'Unit testing'),
        ('Unit testing', 'Unit testing Score 1'),
        ('Unit testing', 'Unit testing Score 2'),
        ('General Engineering Skills', 'Reproducible Programming'),
        ('General Engineering Skills', 'Programming Readability'),
        ('Data Management', 'Intuitive Choices'),
        ('Data Management', 'Assertions'),
        ('Data Management', 'Big Data'),
        (
            'General knowledge of DB & Cloud arch.',
            'General knowledge DB Score 1'
        ),
        (
            'General knowledge of DB & Cloud arch.',
            'General knowledge DB Score 2'
        ),
        ('Intuitive Choices', 'Intuitive Choices Score 1'),
        ('Intuitive Choices', 'Intuitive Choices Score 2'),
        ('Assertions', 'Assertions Score 1'),
        ('Assertions', 'Assertions Score 2'),
        ('Big Data', 'Big Data Score 1'),
        ('Big Data', 'Big Data Score 2'),
        ('Equity', 'Equity Score 1'),
        ('Equity', 'Equity Score 1'),
        ('Learn', 'Learn Score 1'),
        ('Learn', 'Learn Score 2'),
    ]

    for start_var, end_var in med_cpts:
        cpts[end_var] = CPT(
            df=create_df_medium(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    hard_cpts = [
        ('Data Management', 'Big Data'),
    ]

    for start_var, end_var in hard_cpts:
        cpts[end_var] = CPT(
            df=create_df_hard(given=start_var, outcome=end_var),
            outcomes=[end_var],
            givens=[start_var]
        )

    levels_df = create_levels_df()

    cpts['Level'] = CPT(
        df=levels_df,
        outcomes=['Level'],
        givens=[
            'Narrative',
            'Visualization',
            'Statistics',
            'Data Management',
            'Equity',
            'General Engineering Skills',
            'Learn',
        ]
    )

    bayesian_network = BayesianNetwork(
        cpts=list(cpts.values()),
        priors=list(priors.values())
    )

    query_3 = Query(
        givens=[
            {'Big Data Score 1': 4},
            {'Narrative Score 1': 4},
            {'Visualization Score 1': 4},
            {'Statistics Score 1': 4},
            {'Equity Score 1': 4},
            {'Learn Score 1': 4},
            {'Use conditional logic': 4},
            {'Use of functions': 4},
        ],
        outcomes=['Level']
    )

    result_3 = VariableElimination(
        network=bayesian_network,
        query=query_3
    ).compute()

    assert result_3.df[result_3.df['Level'] == 4]['value'].iloc[0] > 0.99
