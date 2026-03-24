
import numpy as np
import pandas as pd
from typing import Optional, Union


class Normalizer:
    """
    Handles normalization and transformation of ZooMS feature matrices.
    Designed for use as a component in a processing pipeline.
    """

    @staticmethod
    def tic_normalize(df: pd.DataFrame, target_sum: float = 10000.0) -> pd.DataFrame:
        """
        Apply Total Ion Current (TIC) normalization.
        Each sample (row) will sum to target_sum.

        Args:
            df (pd.DataFrame): Input feature matrix (samples x bins).
            target_sum (float): The value each row should sum to.

        Returns:
            pd.DataFrame: TIC normalized matrix.
        """
        # Calculate row sums
        row_sums = df.sum(axis=1)
        
        # Avoid division by zero for empty rows
        row_sums = row_sums.replace(0, 1)
        
        # Scale each row: (value / sum) * target_sum
        normalized_df = df.div(row_sums, axis=0) * target_sum
        return normalized_df

    @staticmethod
    def log_transform(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transformation to squash the dynamic range.

        Args:
            df (pd.DataFrame): Input feature matrix.

        Returns:
            pd.DataFrame: Log-transformed matrix.
        """
        return np.log1p(df)

    @staticmethod
    def binary_transform(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """
        Convert intensities to binary presence (1) or absence (0).

        Args:
            df (pd.DataFrame): Input feature matrix.
            threshold (float): Intensity threshold for 'presence'.

        Returns:
            pd.DataFrame: Binary matrix.
        """
        return (df > threshold).astype(int)
