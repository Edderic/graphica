import pandas as pd
import numpy as np

from ..linx.data import ParquetData, InMemoryData
from ..linx.ds import ConditionalProbabilityTable as CPT
from .conftest import clean_tmp, get_tmp_path


def test_cpt_sample_prior():
    """Test sampling from a CPT with no given variables (prior distribution)."""
    clean_tmp()
    
    # Create a simple prior CPT
    df = pd.DataFrame([
        {'X': 0, 'value': 0.3},
        {'X': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['X']
    )
    
    # Sample multiple times and check that we get both values
    samples = []
    for _ in range(100):
        result = cpt.sample()
        samples.append(result['X'])
    
    # Check that we get both values (0 and 1)
    unique_values = set(samples)
    assert 0 in unique_values
    assert 1 in unique_values


def test_cpt_sample_conditional():
    """Test sampling from a CPT with given variables."""
    clean_tmp()
    
    # Create a conditional CPT: P(Y|X)
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.8},
        {'X': 0, 'Y': 1, 'value': 0.2},
        {'X': 1, 'Y': 0, 'value': 0.3},
        {'X': 1, 'Y': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test sampling with X=0
    samples_x0 = []
    for _ in range(50):
        result = cpt.sample(given_values={'X': 0})
        samples_x0.append(result['Y'])
    
    # Test sampling with X=1
    samples_x1 = []
    for _ in range(50):
        result = cpt.sample(given_values={'X': 1})
        samples_x1.append(result['Y'])
    
    # Check that we get valid values
    assert all(val in [0, 1] for val in samples_x0)
    assert all(val in [0, 1] for val in samples_x1)


def test_cpt_sample_multiple_outcomes():
    """Test sampling from a CPT with multiple outcome variables."""
    clean_tmp()
    
    # Create a CPT with multiple outcomes: P(Y,Z|X)
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'Z': 0, 'value': 0.4},
        {'X': 0, 'Y': 0, 'Z': 1, 'value': 0.1},
        {'X': 0, 'Y': 1, 'Z': 0, 'value': 0.2},
        {'X': 0, 'Y': 1, 'Z': 1, 'value': 0.3},
        {'X': 1, 'Y': 0, 'Z': 0, 'value': 0.1},
        {'X': 1, 'Y': 0, 'Z': 1, 'value': 0.3},
        {'X': 1, 'Y': 1, 'Z': 0, 'value': 0.2},
        {'X': 1, 'Y': 1, 'Z': 1, 'value': 0.4}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y', 'Z'],
        givens=['X']
    )
    
    # Test sampling
    result = cpt.sample(given_values={'X': 0})
    
    # Check that both outcome variables are present
    assert 'Y' in result
    assert 'Z' in result
    assert result['Y'] in [0, 1]
    assert result['Z'] in [0, 1]


def test_cpt_sample_missing_given():
    """Test that sampling fails when given variables are missing."""
    clean_tmp()
    
    # Create a conditional CPT: P(Y|X)
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.8},
        {'X': 0, 'Y': 1, 'value': 0.2},
        {'X': 1, 'Y': 0, 'value': 0.3},
        {'X': 1, 'Y': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test that sampling fails when X is not provided
    try:
        cpt.sample()  # Should fail because X is required
        assert False, "Expected ValueError when given variable is missing"
    except ValueError as e:
        assert "Given variables ['X'] are required but no given_values provided" in str(e)


def test_cpt_sample_no_matching_rows():
    """Test that sampling fails when no rows match the given values."""
    clean_tmp()
    
    # Create a conditional CPT: P(Y|X)
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.8},
        {'X': 0, 'Y': 1, 'value': 0.2},
        {'X': 1, 'Y': 0, 'value': 0.3},
        {'X': 1, 'Y': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test that sampling fails when X has an invalid value
    try:
        cpt.sample(given_values={'X': 2})  # X=2 doesn't exist in the data
        assert False, "Expected ValueError when no rows match"
    except ValueError as e:
        assert "No matching rows found" in str(e)


def test_cpt_with_table_parameter():
    """Test creating CPT with table parameter."""
    # Create a CPT using the table parameter
    cpt = CPT(
        table=[
            {'X': 0, 'Y': 0, 'value': 0.25},
            {'X': 0, 'Y': 1, 'value': 0.75},
            {'X': 1, 'Y': 0, 'value': 0.6},
            {'X': 1, 'Y': 1, 'value': 0.4},
        ],
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test that the CPT was created correctly
    assert cpt.get_outcomes() == ['Y']
    assert cpt.get_givens() == ['X']
    
    # Test sampling
    result = cpt.sample(given_values={'X': 0})
    assert 'Y' in result
    assert result['Y'] in [0, 1]


def test_cpt_table_vs_data_equivalence():
    """Test that table and data parameters create equivalent CPTs."""
    # Create CPT using table parameter
    cpt_table = CPT(
        table=[
            {'X': 0, 'Y': 0, 'value': 0.25},
            {'X': 0, 'Y': 1, 'value': 0.75},
            {'X': 1, 'Y': 0, 'value': 0.6},
            {'X': 1, 'Y': 1, 'value': 0.4},
        ],
        outcomes=['Y'],
        givens=['X']
    )
    
    # Create equivalent CPT using data parameter
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])
    
    from ..linx.data import InMemoryData
    cpt_data = CPT(
        data=InMemoryData(df),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test that both CPTs have the same structure
    assert cpt_table.get_outcomes() == cpt_data.get_outcomes()
    assert cpt_table.get_givens() == cpt_data.get_givens()
    
    # Test that both CPTs produce similar sampling results
    samples_table = []
    samples_data = []
    for _ in range(1000):  # Increase sample size for more stable comparison
        samples_table.append(cpt_table.sample(given_values={'X': 0})['Y'])
        samples_data.append(cpt_data.sample(given_values={'X': 0})['Y'])
    
    # The means should be similar (increased tolerance for randomness)
    mean_table = np.array(samples_table).mean()
    mean_data = np.array(samples_data).mean()
    assert abs(mean_table - mean_data) < 0.2  # Increased tolerance


def test_cpt_invalid_parameters():
    """Test that invalid parameter combinations raise appropriate errors."""
    # Test missing both data and table
    try:
        CPT(outcomes=['Y'], givens=['X'])
        assert False, "Expected ValueError when neither data nor table is provided"
    except ValueError as e:
        assert "Either 'data' or 'table' parameter must be provided" in str(e)
    
    # Test providing both data and table
    try:
        CPT(
            data=InMemoryData(pd.DataFrame([{'X': 0, 'value': 1.0}])),
            table=[{'X': 0, 'value': 1.0}],
            outcomes=['X']
        )
        assert False, "Expected ValueError when both data and table are provided"
    except ValueError as e:
        assert "Cannot specify both 'data' and 'table' parameters" in str(e)
    
    # Test missing outcomes
    try:
        CPT(table=[{'X': 0, 'value': 1.0}])
        assert False, "Expected ValueError when outcomes is missing"
    except ValueError as e:
        assert "outcomes parameter is required" in str(e) 