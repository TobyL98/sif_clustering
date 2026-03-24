
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.phase4_clustering import run_phase4

def test_run_phase4(tmp_path):
    """Verify that Phase 4 orchestration generates the results file."""
    # Create mock results directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Mock binary feature matrix (10 samples, 5 bins)
    X_df = pd.DataFrame(
        np.random.randint(0, 2, size=(10, 5)),
        index=[f"S{i}" for i in range(10)],
        columns=[f"bin{i}" for i in range(5)]
    )
    X_df.to_csv(results_dir / "feature_matrix_binary.csv")
    
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    try:
        results_df = run_phase4(
            feature_path="results/feature_matrix_binary.csv",
            n_clusters=2
        )
        
        # Verify output files
        assert Path("results/clustering_results.csv").exists()
        assert "KMeans_Cluster" in results_df.columns
        assert "Hierarchical_Cluster" in results_df.columns
        assert len(results_df) == 10
        
    finally:
        os.chdir(original_cwd)
