
import pytest
import pandas as pd
from src.phase2_normalization import run_phase2
from pathlib import Path
import os

def test_run_phase2(tmp_path):
    """Verify that Phase 2 orchestration generates all 3 expected output files."""
    # Create a mock Phase 1 file using Path
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    input_file = results_dir / "feature_matrix_phase1.csv"
    
    mock_df = pd.DataFrame({
        "bin1": [10.0, 1000.0],
        "bin2": [20.0, 2000.0],
    }, index=["S1", "S2"])
    mock_df.to_csv(input_file)
    
    # Change directory to tmp_path so "results/" is found relative to it
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    try:
        # Pass path relative to new CWD
        run_phase2("results/feature_matrix_phase1.csv")
        
        # Verify all 3 files exist using Path
        assert (Path("results") / "feature_matrix_tic.csv").exists()
        assert (Path("results") / "feature_matrix_tic_log.csv").exists()
        assert (Path("results") / "feature_matrix_binary.csv").exists()
        
        # Load one to check structure
        binary_df = pd.read_csv(Path("results") / "feature_matrix_binary.csv", index_col=0)
        assert binary_df.iloc[0, 0] == 1
        assert binary_df.shape == mock_df.shape
        
    finally:
        os.chdir(original_cwd)
