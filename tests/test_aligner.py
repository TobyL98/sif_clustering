
import numpy as np
import pandas as pd
import pytest
from src.aligner import PeakAligner

def test_aligner_initialization():
    """Test that aligner initializes with correct bin count."""
    bin_size = 0.4
    mz_min, mz_max = 800, 3500
    aligner = PeakAligner(bin_size=bin_size, mz_range=(mz_min, mz_max))
    
    # Expected number of bins: (3500 - 800) / 0.4 = 6750
    expected_bins = int((mz_max - mz_min) / bin_size)
    assert len(aligner.get_bin_centers()) == expected_bins

def test_align_single_peak():
    """Test that a single peak is placed in the correct bin."""
    aligner = PeakAligner(bin_size=1.0, mz_range=(800, 1000))
    # Peak at 800.5 should be in the first bin (800-801)
    peaks = pd.DataFrame({"mz": [800.5], "intensity": [100.0]})
    vector = aligner.align_sample(peaks)
    
    assert vector[0] == 100.0
    assert np.sum(vector) == 100.0
    assert vector[1] == 0.0

def test_align_multiple_peaks_in_same_bin():
    """Test that multiple peaks in the same bin are summed."""
    aligner = PeakAligner(bin_size=1.0, mz_range=(800, 1000))
    peaks = pd.DataFrame({"mz": [800.1, 800.9], "intensity": [50.0, 60.0]})
    vector = aligner.align_sample(peaks)
    
    assert vector[0] == 110.0

def test_align_peaks_out_of_range():
    """Test that peaks out of range are ignored."""
    aligner = PeakAligner(bin_size=1.0, mz_range=(800, 1000))
    peaks = pd.DataFrame({"mz": [799.0, 1001.0], "intensity": [50.0, 60.0]})
    vector = aligner.align_sample(peaks)
    
    assert np.sum(vector) == 0.0
