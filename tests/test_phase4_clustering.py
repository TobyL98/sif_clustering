
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.phase4_clustering import run_phase4

def test_run_phase4_multi(tmp_path):
    """Verify that Phase 4 generates all 5 expected clustering result columns."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create mock binary matrix
    X_df = pd.DataFrame(
        np.random.randint(0, 2, size=(20, 10)),
        index=[f"S{i}" for i in range(20)],
        columns=[f"bin{i}" for i in range(10)]
    )
    X_df.to_csv(results_dir / "feature_matrix_binary.csv")
    
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    try:
        results_df = run_phase4(
            feature_path="results/feature_matrix_binary.csv",
            n_clusters=2
        )
        
        # Verify result file
        assert Path("results/clustering_results.csv").exists()
        
        # Check all methods are present (The names returned by the methods in ClusteringModel)
        expected_cols = [
            "KMeans_Cluster", 
            "Hierarchical_ward_euclidean_Cluster", 
            "Hierarchical_average_jaccard_Cluster", 
            "Spectral_Cluster", 
            "HDBSCAN_Cluster"
        ]
        for col in expected_cols:
            assert col in results_df.columns
        
        assert len(results_df) == 20
        
    finally:
        os.chdir(original_cwd)
