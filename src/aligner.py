
import numpy as np
import pandas as pd
from typing import Tuple


class PeakAligner:
    """
    Handles binning and alignment of mass spectrometry peaks.
    """

    def __init__(self, bin_size: float = 0.4, mz_range: Tuple[float, float] = (800.0, 3500.0)):
        """
        Initialize the PeakAligner.

        Args:
            bin_size (float): The width of each m/z bin in Daltons.
            mz_range (Tuple[float, float]): The min and max m/z to consider.
        """
        self.bin_size = bin_size
        self.mz_range = mz_range
        # np.arange(start, stop, step)
        self.bins = np.arange(mz_range[0], mz_range[1] + bin_size, bin_size)

    def align_sample(self, peaks: pd.DataFrame) -> np.ndarray:
        """
        Bins a single sample's peaks into the predefined m/z grid.

        Args:
            peaks (pd.DataFrame): DataFrame with 'mz' and 'intensity' columns.

        Returns:
            np.ndarray: Binned intensities (feature vector).
        """
        # Filter peaks within range
        mask = (peaks["mz"] >= self.mz_range[0]) & (peaks["mz"] <= self.mz_range[1])
        filtered_peaks = peaks[mask]

        # Initialize feature vector (number of bins is len(self.bins) - 1)
        feature_vector = np.zeros(len(self.bins) - 1)

        if filtered_peaks.empty:
            return feature_vector

        # Digitization: find which bin each mz belongs to
        # np.digitize returns indices 1 to len(bins) for values inside range
        indices = np.digitize(filtered_peaks["mz"].values, self.bins)
        
        # We want to create a vector where we sum intensities in each bin
        # Bin index 1 corresponds to bins[0] to bins[1]
        for idx, intensity in zip(indices, filtered_peaks["intensity"].values):
            # Only process if idx is within the valid range of feature_vector
            if 1 <= idx < len(self.bins):
                feature_vector[idx - 1] += intensity
                
        return feature_vector

    def get_bin_centers(self) -> np.ndarray:
        """
        Returns the centers of the bins for labeling.

        Returns:
            np.ndarray: Centers of m/z bins.
        """
        return self.bins[:-1] + self.bin_size / 2
