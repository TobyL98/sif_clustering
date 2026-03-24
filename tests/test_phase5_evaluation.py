
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.phase5_evaluation import run_phase5

def test_run_phase5(tmp_path):
    """Verify that Phase 5 orchestration generates all expected output files."""
    # Create mock results directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Mock clustering results (4 samples)
    cluster_df = pd.DataFrame({
        "KMeans_Cluster": [0, 0, 1, 1],
        "Hierarchical_Cluster": [1, 1, 0, 0]
    }, index=["S1", "S2", "S3", "S4"])
    cluster_df.to_csv(results_dir / "clustering_results.csv")
    
    # Mock metadata
    meta_df = pd.DataFrame({
        "Sample Name": ["S1", "S2", "S3", "S4"],
        "Correct ID": ["FamilyA", "FamilyA", "FamilyB", "FamilyB"]
    })
    meta_df.to_csv(results_dir / "sample_metadata_phase1.csv", index=False)
    
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    try:
        metrics_df = run_phase5(
            clustering_path="results/clustering_results.csv",
            metadata_path="results/sample_metadata_phase1.csv"
        )
        
        # Verify output files exist
        assert Path("results/evaluation_metrics.csv").exists()
        assert Path("results/contingency_kmeans.csv").exists()
        assert Path("results/contingency_hierarchical.csv").exists()
        
        # Check metrics for perfect clustering
        assert metrics_df.loc["K-Means", "ARI"] == 1.0
        assert metrics_df.loc["Hierarchical", "V-Measure"] == 1.0
        
    finally:
        os.chdir(original_cwd)
