
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.phase3_visualization import run_phase3

def test_run_phase3(tmp_path):
    """Verify that Phase 3 orchestration generates the plot file."""
    # Create mock results directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Mock binary feature matrix (5 samples, 10 bins)
    X_df = pd.DataFrame(
        np.random.randint(0, 2, size=(5, 10)),
        index=[f"S{i}" for i in range(5)],
        columns=[f"bin{i}" for i in range(10)]
    )
    X_df.to_csv(results_dir / "feature_matrix_binary.csv")
    
    # Mock metadata
    meta_df = pd.DataFrame({
        "Sample Name": [f"S{i}" for i in range(5)],
        "Correct ID": ["FamilyA", "FamilyA", "FamilyB", "FamilyB", "FamilyA"]
    })
    meta_df.to_csv(results_dir / "sample_metadata_phase1.csv", index=False)
    
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    try:
        pca_df, tsne_df = run_phase3(
            feature_path="results/feature_matrix_binary.csv",
            metadata_path="results/sample_metadata_phase1.csv"
        )
        
        # Verify output files
        assert (Path("results/plots") / "dim_reduction_binary.png").exists()
        assert len(pca_df) == 5
        assert len(tsne_df) == 5
        
    finally:
        os.chdir(original_cwd)
