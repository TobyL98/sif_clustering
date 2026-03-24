
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import Optional, Dict


class ClusteringModel:
    """
    Handles clustering algorithms for ZooMS feature matrices.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the clustering module.

        Args:
            random_state (int): Seed for reproducibility.
        """
        self.random_state = random_state

    def run_kmeans(self, df: pd.DataFrame, n_clusters: int) -> pd.Series:
        """
        Run K-Means clustering.

        Args:
            df (pd.DataFrame): Feature matrix.
            n_clusters (int): Number of clusters.

        Returns:
            pd.Series: Cluster labels indexed by Sample Name.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(df)
        return pd.Series(labels, index=df.index, name="KMeans_Cluster")

    def run_hierarchical(self, df: pd.DataFrame, n_clusters: int) -> pd.Series:
        """
        Run Agglomerative Hierarchical Clustering (Ward's linkage).

        Args:
            df (pd.DataFrame): Feature matrix.
            n_clusters (int): Number of clusters.

        Returns:
            pd.Series: Cluster labels indexed by Sample Name.
        """
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(df)
        return pd.Series(labels, index=df.index, name="Hierarchical_Cluster")
