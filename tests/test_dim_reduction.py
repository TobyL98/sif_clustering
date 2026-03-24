
import pytest
import pandas as pd
import numpy as np
from src.dim_reduction import DimensionalityReducer

@pytest.fixture
def mock_data():
    """Create a mock 5x10 feature matrix."""
    X = np.random.rand(5, 10)
    return pd.DataFrame(X, index=[f"S{i}" for i in range(5)])

def test_pca_output_shape(mock_data):
    """Verify that PCA returns the requested number of columns."""
    reducer = DimensionalityReducer()
    pca_df = reducer.run_pca(mock_data, n_components=2)
    
    assert pca_df.shape == (5, 2)
    assert pca_df.index.tolist() == mock_data.index.tolist()
    assert list(pca_df.columns) == ["PC1", "PC2"]

def test_tsne_output_shape(mock_data):
    """Verify that t-SNE returns the requested number of columns."""
    reducer = DimensionalityReducer()
    tsne_df = reducer.run_tsne(mock_data, n_components=2)
    
    assert tsne_df.shape == (5, 2)
    assert tsne_df.index.tolist() == mock_data.index.tolist()
    assert list(tsne_df.columns) == ["tSNE1", "tSNE2"]

def test_determinism(mock_data):
    """Verify that results are identical when using the same random_state."""
    reducer1 = DimensionalityReducer(random_state=42)
    reducer2 = DimensionalityReducer(random_state=42)
    
    pca1 = reducer1.run_pca(mock_data)
    pca2 = reducer2.run_pca(mock_data)
    
    pd.testing.assert_frame_equal(pca1, pca2)
    
    # tsne is also randomized
    tsne1 = reducer1.run_tsne(mock_data)
    tsne2 = reducer2.run_tsne(mock_data)
    pd.testing.assert_frame_equal(tsne1, tsne2)
