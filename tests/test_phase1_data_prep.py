
import pytest
import pandas as pd
import numpy as np
from src.phase1_data_prep import run_phase1
from pathlib import Path

def test_run_phase1(tmp_path):
    """
    Test that Phase 1 orchestration correctly builds a matrix.
    """
    # Create a mock structure
    root = tmp_path / "project"
    peak_dir = root / "peaklist_inputs"
    peak_dir.mkdir(parents=True)
    
    # Peak list file with 2 peaks: 800.5 and 801.5
    peak_content = "800.5\t100.0\n801.5\t200.0"
    (peak_dir / "S1_peaklist.txt").write_text(peak_content)
    (peak_dir / "S2_peaklist.txt").write_text(peak_content)
    
    # Metadata
    csv_path = root / "metadata.csv"
    pd.DataFrame({
        "Sample Name": ["S1", "S2"],
        "Correct ID": ["FamilyA", "FamilyA"]
    }).to_csv(csv_path, index=False)
    
    # Patch DataLoader root_dir to use our tmp_path
    # Instead of monkeypatching the class, we change directory to tmp_path
    # so DataLoader's default '.' works correctly.
    import os
    original_cwd = os.getcwd()
    os.chdir(root)
    
    try:
        X_df, mapping_df = run_phase1("metadata.csv")
        
        # Check X_df
        assert isinstance(X_df, pd.DataFrame)
        assert len(X_df) == 2
        assert X_df.index.tolist() == ["S1", "S2"]
        
        # Check mapping_df
        assert len(mapping_df) == 2
        assert mapping_df["Sample Name"].tolist() == ["S1", "S2"]
        
        # Check feature values (using default 0.4 bin size)
        # First bin 800-800.4
        # Second bin 800.4-800.8 (contains 800.5)
        # 801.5 will be in a later bin
        assert X_df.iloc[0].sum() == 300.0 # Total intensity
        
        # Verify result files exist
        assert Path("results/feature_matrix_phase1.csv").exists()
        assert Path("results/sample_metadata_phase1.csv").exists()
        
    finally:
        os.chdir(original_cwd)
