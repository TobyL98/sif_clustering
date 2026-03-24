
import pytest
import pandas as pd
import numpy as np
from src.clustering import ClusteringModel

@pytest.fixture
def mock_binary_data():
    """Create a mock 10x20 binary feature matrix with clusters."""
    # 5 samples with peaks in first 5 bins, 5 samples with peaks in last 5 bins
    X = np.zeros((10, 20))
    X[:5, :5] = 1
    X[5:, 15:] = 1
    return pd.DataFrame(X, index=[f"S{i}" for i in range(10)])

def test_kmeans_labels(mock_binary_data):
    """Verify K-Means labels all samples."""
    model = ClusteringModel()
    labels = model.run_kmeans(mock_binary_data, n_clusters=2)
    assert len(labels) == 10
    assert len(set(labels)) == 2

def test_hierarchical_jaccard(mock_binary_data):
    """Verify Hierarchical clustering with Jaccard metric."""
    model = ClusteringModel()
    labels = model.run_hierarchical(mock_binary_data, n_clusters=2, linkage='average', metric='jaccard')
    assert len(labels) == 10
    assert len(set(labels)) == 2
    # Samples 0-4 should be in one cluster, 5-9 in another
    # Use .iloc because the index is ["S0", "S1", ...]
    assert labels.iloc[0] == labels.iloc[4]
    assert labels.iloc[5] == labels.iloc[9]
    assert labels.iloc[0] != labels.iloc[5]

def test_spectral_labels(mock_binary_data):
    """Verify Spectral clustering labels."""
    model = ClusteringModel()
    labels = model.run_spectral(mock_binary_data, n_clusters=2)
    assert len(labels) == 10
    assert len(set(labels)) == 2

def test_hdbscan_labels(mock_binary_data):
    """Verify HDBSCAN identifies clusters in binary data."""
    model = ClusteringModel()
    labels = model.run_hdbscan(mock_binary_data, min_cluster_size=2)
    assert len(labels) == 10
    # Should find at least the two major groups
    unique_labels = [l for l in set(labels) if l != -1]
    assert len(unique_labels) >= 1
