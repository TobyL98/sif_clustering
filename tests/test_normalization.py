
import pandas as pd
import numpy as np
import pytest
from src.normalization import Normalizer

@pytest.fixture
def mock_matrix():
    """Create a sample intensity matrix."""
    return pd.DataFrame({
        "bin1": [10.0, 1000.0, 0.0],
        "bin2": [20.0, 2000.0, 50.0],
    }, index=["S1", "S2", "S3"])

def test_tic_normalization(mock_matrix):
    """Verify that rows sum to the target value after TIC."""
    target = 10000.0
    normalized = Normalizer.tic_normalize(mock_matrix, target_sum=target)
    
    # Check that each row sums to 10000.0
    row_sums = normalized.sum(axis=1)
    for s in row_sums:
        assert pytest.approx(s) == target
        
    # Check that relative relationships are preserved
    # In S1, bin2 is 2x bin1. This should still hold.
    assert pytest.approx(normalized.loc["S1", "bin2"]) == normalized.loc["S1", "bin1"] * 2

def test_log_transformation(mock_matrix):
    """Verify that log1p squashes range and handles 0 correctly."""
    transformed = Normalizer.log_transform(mock_matrix)
    
    # 0 should stay 0 (log1p(0) = 0)
    assert transformed.loc["S3", "bin1"] == 0.0
    
    # Check a value
    expected_s1_bin1 = np.log1p(10.0)
    assert pytest.approx(transformed.loc["S1", "bin1"]) == expected_s1_bin1

def test_binary_transformation(mock_matrix):
    """Verify that values are converted to 0 or 1."""
    binary = Normalizer.binary_transform(mock_matrix)
    
    assert set(binary.values.flatten()).issubset({0, 1})
    assert binary.loc["S1", "bin1"] == 1
    assert binary.loc["S3", "bin1"] == 0
    assert binary.index.tolist() == mock_matrix.index.tolist()
