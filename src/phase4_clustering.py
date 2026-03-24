
import pandas as pd
from pathlib import Path
from src.clustering import ClusteringModel


def run_phase4(
    feature_path: str = "results/feature_matrix_binary.csv",
    n_clusters: int = 10 # Number of families in ground truth
):
    """
    Executes Phase 4 of the development plan:
    - Load binary feature matrix
    - Perform K-Means clustering
    - Perform Hierarchical clustering
    - Save cluster assignments
    """
    print(f"Starting Phase 4: Clustering on {feature_path}...")
    
    # 1. Load Data
    X_df = pd.read_csv(Path(feature_path), index_col=0)
    
    model = ClusteringModel()
    
    # 2. Run K-Means
    print(f"Running K-Means (n_clusters={n_clusters})...")
    kmeans_labels = model.run_kmeans(X_df, n_clusters=n_clusters)
    
    # 3. Run Hierarchical
    print(f"Running Hierarchical (n_clusters={n_clusters})...")
    agg_labels = model.run_hierarchical(X_df, n_clusters=n_clusters)
    
    # 4. Combine and Save Results
    results_df = pd.DataFrame({
        "KMeans_Cluster": kmeans_labels,
        "Hierarchical_Cluster": agg_labels
    })
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "clustering_results.csv"
    results_df.to_csv(results_file)
    
    print(f"Phase 4 complete. Cluster assignments saved to {results_file}")
    return results_df


if __name__ == "__main__":
    run_phase4()
