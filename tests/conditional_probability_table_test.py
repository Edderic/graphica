import pytest
import numpy as np
import pandas as pd

from ..graphica.data import ParquetData
from ..graphica.ds import ConditionalProbabilityTable as CPT
from ..graphica.random.random_variable import RandomVariable
from .conftest import clean_tmp, get_tmp_path


def test_cpt_inherits_from_random_variable():
    """Test that CPT inherits from RandomVariable."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'value': 0.3},
        {'X': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['X']
    )
    
    assert isinstance(cpt, RandomVariable)
    assert isinstance(cpt, CPT)


def test_cpt_pdf_method():
    """Test the pdf method of CPT."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test with dict input
    prob = cpt.pdf({'Y': 1}, X=0)
    assert prob == 0.75
    
    # Test with array input
    prob = cpt.pdf([1], X=0)
    assert prob == 0.75
    
    # Test with non-existent values
    prob = cpt.pdf({'Y': 2}, X=0)
    assert prob == 0.0


def test_cpt_logpdf_method():
    """Test the logpdf method of CPT."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'value': 0.3},
        {'X': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['X']
    )
    
    # Test valid probability
    log_prob = cpt.logpdf({'X': 1})
    expected_log_prob = np.log(0.7)
    assert abs(log_prob - expected_log_prob) < 1e-10
    
    # Test zero probability
    log_prob = cpt.logpdf({'X': 2})
    assert log_prob == -np.inf


def test_cpt_sample_method():
    """Test the sample method of CPT."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'value': 0.3},
        {'X': 1, 'value': 0.7}
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['X']
    )
    
    # Test single sample
    sample = cpt.sample()
    assert isinstance(sample, dict)
    assert 'X' in sample
    assert sample['X'] in [0, 1]
    
    # Test multiple samples
    samples = cpt.sample(size=10)
    assert len(samples) == 10
    for sample in samples:
        assert isinstance(sample, dict)
        assert 'X' in sample
        assert sample['X'] in [0, 1]


def test_cpt_sample_with_conditioning():
    """Test sampling from CPT with conditioning."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test sampling with conditioning
    sample = cpt.sample(X=0)
    assert isinstance(sample, dict)
    assert 'Y' in sample
    assert sample['Y'] in [0, 1]
    
    # Test multiple samples with conditioning
    samples = cpt.sample(size=5, X=1)
    assert len(samples) == 5
    for sample in samples:
        assert isinstance(sample, dict)
        assert 'Y' in sample
        assert sample['Y'] in [0, 1]


def test_cpt_backward_compatibility():
    """Test that existing CPT functionality still works."""
    clean_tmp()
    
    df = pd.DataFrame([
        {'X': 0, 'Y': 0, 'value': 0.25},
        {'X': 0, 'Y': 1, 'value': 0.75},
        {'X': 1, 'Y': 0, 'value': 0.6},
        {'X': 1, 'Y': 1, 'value': 0.4},
    ])
    
    cpt = CPT(
        ParquetData(df, storage_folder=get_tmp_path()),
        outcomes=['Y'],
        givens=['X']
    )
    
    # Test original sample method
    sample = cpt.sample_with_given_values({'X': 0})
    assert isinstance(sample, dict)
    assert 'Y' in sample
    assert sample['Y'] in [0, 1]
    
    # Test other original methods
    assert cpt.get_outcomes() == ['Y']
    assert cpt.get_givens() == ['X']
    assert cpt.get_data() is not None 