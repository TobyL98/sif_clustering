
import pytest
import pandas as pd
import numpy as np
from src.clustering import ClusteringModel

@pytest.fixture
def mock_data():
    """Create a mock 10x2 feature matrix with clear clusters."""
    # 5 samples near (0,0), 5 samples near (10,10)
    X = np.vstack([
        np.random.normal(0, 1, (5, 2)),
        np.random.normal(10, 1, (5, 2))
    ])
    return pd.DataFrame(X, index=[f"S{i}" for i in range(10)])

def test_kmeans_labels(mock_data):
    """Verify K-Means labels all samples and returns the correct index."""
    model = ClusteringModel()
    labels = model.run_kmeans(mock_data, n_clusters=2)
    
    assert len(labels) == 10
    assert labels.index.tolist() == mock_data.index.tolist()
    assert len(set(labels)) == 2
    assert labels.name == "KMeans_Cluster"

def test_hierarchical_labels(mock_data):
    """Verify Hierarchical labels all samples and returns the correct index."""
    model = ClusteringModel()
    labels = model.run_hierarchical(mock_data, n_clusters=2)
    
    assert len(labels) == 10
    assert labels.index.tolist() == mock_data.index.tolist()
    assert len(set(labels)) == 2
    assert labels.name == "Hierarchical_Cluster"

def test_reproducibility(mock_data):
    """Verify that K-Means is deterministic with a fixed random_state."""
    model1 = ClusteringModel(random_state=42)
    model2 = ClusteringModel(random_state=42)
    
    labels1 = model1.run_kmeans(mock_data, n_clusters=2)
    labels2 = model2.run_kmeans(mock_data, n_clusters=2)
    
    pd.testing.assert_series_equal(labels1, labels2)
