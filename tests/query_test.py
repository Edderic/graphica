from ..linx.ds import Query


def test_query_get_outcome_variables():
    query = Query(
        outcomes=['X', {'Y': lambda df: df['Y'] > 5}]
    )

    outcome_vars = query.get_outcome_variables()
    assert set(outcome_vars) == {'X', 'Y'}


def test_query_get_given_variables():
    query = Query(
        outcomes=['X', {'Y': lambda df: df['Y'] > 5}],
        givens=['Z', {'A': lambda df: df['A'] > 5}]
    )

    given_vars = query.get_given_variables()
    assert set(given_vars) == {'Z', 'A'}


def test_query_get_filters():
    query = Query(
        outcomes=['X', {'Y': lambda df: df['Y'] > 5}],
        givens=['Z', {'A': lambda df: df['A'] > 5}]
    )

    filters = query.get_filters()

    keys = [list(f.keys())[0] for f in filters]
    assert 'Y' in keys
    assert 'A' in keys
    assert 'X' not in keys
    assert 'Z' not in keys


def test_query_get_filters_without_givens():
    query = Query(
        outcomes=['X', {'Y': lambda df: df['Y'] > 5}],
    )

    filters = query.get_filters()

    keys = [list(f.keys())[0] for f in filters]
    assert 'Y' in keys
    assert 'A' not in keys
    assert 'X' not in keys
    assert 'Z' not in keys
