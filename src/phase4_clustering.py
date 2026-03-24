
import pandas as pd
from pathlib import Path
from src.clustering import ClusteringModel


def run_phase4(
    feature_path: str = "results/feature_matrix_binary.csv",
    n_clusters: int = 10 
):
    """
    Executes Phase 4: Clustering with multiple methods.
    """
    print(f"Starting Phase 4: Extended Clustering on {feature_path}...")
    
    X_df = pd.read_csv(Path(feature_path), index_col=0)
    model = ClusteringModel()
    results = {}
    
    # Baseline
    print("Running K-Means...")
    results["KMeans"] = model.run_kmeans(X_df, n_clusters)
    
    # Hierarchical (Ward/Euclidean)
    print("Running Hierarchical (Ward/Euclidean)...")
    results["Hierarchical_Ward"] = model.run_hierarchical(X_df, n_clusters, linkage='ward')
    
    # Hierarchical (Average/Jaccard) - Ideal for Binary
    print("Running Hierarchical (Average/Jaccard)...")
    results["Hierarchical_Jaccard"] = model.run_hierarchical(X_df, n_clusters, linkage='average', metric='jaccard')
    
    # Spectral
    print("Running Spectral Clustering...")
    results["Spectral"] = model.run_spectral(X_df, n_clusters)
    
    # HDBSCAN
    print("Running HDBSCAN...")
    results["HDBSCAN"] = model.run_hdbscan(X_df)
    
    # Combine
    results_df = pd.concat(results.values(), axis=1)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "clustering_results.csv")
    
    print(f"Phase 4 complete. Results saved with {len(results)} methods.")
    return results_df


if __name__ == "__main__":
    run_phase4()
