
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from hdbscan import HDBSCAN
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
        Run K-Means clustering (Euclidean).

        Args:
            df (pd.DataFrame): Feature matrix.
            n_clusters (int): Number of clusters.

        Returns:
            pd.Series: Cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(df)
        return pd.Series(labels, index=df.index, name="KMeans_Cluster")

    def run_hierarchical(self, df: pd.DataFrame, n_clusters: int, linkage: str = 'ward', metric: str = 'euclidean') -> pd.Series:
        """
        Run Agglomerative Hierarchical Clustering.

        Args:
            df (pd.DataFrame): Feature matrix.
            n_clusters (int): Number of clusters.
            linkage (str): Linkage method ('ward', 'average', 'complete').
            metric (str): Distance metric ('euclidean', 'jaccard', etc.).

        Returns:
            pd.Series: Cluster labels.
        """
        # Note: 'ward' linkage only supports 'euclidean' metric
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
        labels = agg.fit_predict(df)
        return pd.Series(labels, index=df.index, name=f"Hierarchical_{linkage}_{metric}_Cluster")

    def run_spectral(self, df: pd.DataFrame, n_clusters: int) -> pd.Series:
        """
        Run Spectral Clustering (Handles non-linear structures).

        Args:
            df (pd.DataFrame): Feature matrix.
            n_clusters (int): Number of clusters.

        Returns:
            pd.Series: Cluster labels.
        """
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            affinity='nearest_neighbors'
        )
        labels = spectral.fit_predict(df)
        return pd.Series(labels, index=df.index, name="Spectral_Cluster")

    def run_hdbscan(self, df: pd.DataFrame, min_cluster_size: int = 5) -> pd.Series:
        """
        Run HDBSCAN (Density-based, identifies noise as -1).

        Args:
            df (pd.DataFrame): Feature matrix.
            min_cluster_size (int): Minimum size of a cluster.

        Returns:
            pd.Series: Cluster labels.
        """
        # HDBSCAN works well with Jaccard for binary data
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric='jaccard')
        labels = hdb.fit_predict(df.astype(bool))
        return pd.Series(labels, index=df.index, name="HDBSCAN_Cluster")
