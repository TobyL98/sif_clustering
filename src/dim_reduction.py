
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple


class DimensionalityReducer:
    """
    Handles dimensionality reduction for ZooMS feature matrices.
    Ensures reproducibility with fixed random states.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the reducer.

        Args:
            random_state (int): Seed for randomized algorithms.
        """
        self.random_state = random_state

    def run_pca(self, df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """
        Run Principal Component Analysis (PCA).

        Args:
            df (pd.DataFrame): Input feature matrix.
            n_components (int): Number of components to return.

        Returns:
            pd.DataFrame: PCA components with original index.
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        components = pca.fit_transform(df)
        
        cols = [f"PC{i+1}" for i in range(n_components)]
        return pd.DataFrame(components, index=df.index, columns=cols)

    def run_tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> pd.DataFrame:
        """
        Run t-Distributed Stochastic Neighbor Embedding (t-SNE).

        Args:
            df (pd.DataFrame): Input feature matrix.
            n_components (int): Number of components to return.
            perplexity (float): Balancing local and global aspects of the data.

        Returns:
            pd.DataFrame: t-SNE components with original index.
        """
        # Perplexity must be less than the number of samples
        n_samples = df.shape[0]
        actual_perplexity = min(perplexity, n_samples - 1)
        
        tsne = TSNE(
            n_components=n_components, 
            perplexity=actual_perplexity,
            random_state=self.random_state,
            init='pca',
            learning_rate='auto'
        )
        components = tsne.fit_transform(df)
        
        cols = [f"tSNE{i+1}" for i in range(n_components)]
        return pd.DataFrame(components, index=df.index, columns=cols)
