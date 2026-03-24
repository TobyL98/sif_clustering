
import pytest
import pandas as pd
from src.data_loader import DataLoader
from pathlib import Path

def test_get_sample_mapping(tmp_path):
    """
    Test that sample mapping correctly links names to paths and handles suffixes.
    """
    # Create a mock structure
    root = tmp_path / "project"
    peak_dir = root / "peaklist_inputs"
    peak_dir.mkdir(parents=True)
    
    # Create mock peak files
    (peak_dir / "sample1_peaklist.txt").write_text("800.0\t100")
    (peak_dir / "sample2.txt").write_text("800.0\t100")
    
    # Create mock CSV
    csv_path = root / "metadata.csv"
    pd.DataFrame({
        "Sample Name": ["sample1", "sample2", "missing_sample"]
    }).to_csv(csv_path, index=False)
    
    loader = DataLoader(root_dir=str(root))
    mapping = loader.get_sample_mapping("metadata.csv")
    
    # Should find sample1 and sample2, but not missing_sample
    assert len(mapping) == 2
    assert "sample1" in mapping["Sample Name"].values
    assert "sample2" in mapping["Sample Name"].values
    assert "missing_sample" not in mapping["Sample Name"].values
    
    # Check paths
    assert mapping[mapping["Sample Name"] == "sample1"]["file_path"].iloc[0].endswith("sample1_peaklist.txt")
    assert mapping[mapping["Sample Name"] == "sample2"]["file_path"].iloc[0].endswith("sample2.txt")

def test_load_peaks(tmp_path):
    """
    Test that peaks are loaded correctly from a file.
    """
    peak_file = tmp_path / "test_peaks.txt"
    peak_file.write_text("801.5\t200.0\n802.5\t300.0")
    
    loader = DataLoader()
    peaks = loader.load_peaks(str(peak_file))
    
    assert len(peaks) == 2
    assert peaks.iloc[0]["mz"] == 801.5
    assert peaks.iloc[0]["intensity"] == 200.0
