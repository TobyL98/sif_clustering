
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score, homogeneity_score, completeness_score
from typing import Dict


class Evaluator:
    """
    Handles evaluation of clustering results against ground truth.
    """

    @staticmethod
    def calculate_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard clustering validation metrics.

        Args:
            labels_true (np.ndarray): Ground truth class labels.
            labels_pred (np.ndarray): Predicted cluster labels.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        return {
            "ARI": adjusted_rand_score(labels_true, labels_pred),
            "V-Measure": v_measure_score(labels_true, labels_pred),
            "Homogeneity": homogeneity_score(labels_true, labels_pred),
            "Completeness": completeness_score(labels_true, labels_pred)
        }

    @staticmethod
    def get_cluster_counts(df: pd.DataFrame, cluster_col: str, ground_truth_col: str) -> pd.DataFrame:
        """
        Create a cross-tabulation of clusters vs ground truth classes.
        Useful for identifying which families are mixed in which clusters.

        Args:
            df (pd.DataFrame): DataFrame containing both cluster and truth labels.
            cluster_col (str): Name of the cluster ID column.
            ground_truth_col (str): Name of the ground truth label column.

        Returns:
            pd.DataFrame: Contingency table (cross-tab).
        """
        return pd.crosstab(df[cluster_col], df[ground_truth_col])
