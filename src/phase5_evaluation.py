
import pandas as pd
from pathlib import Path
from src.evaluation import Evaluator


def run_phase5(
    clustering_path: str = "results/clustering_results.csv",
    metadata_path: str = "results/sample_metadata_phase1.csv"
):
    """
    Executes Phase 5 of the development plan:
    - Load clustering results and ground truth metadata
    - Merge results by Sample Name
    - Calculate ARI, V-Measure, Homogeneity, Completeness
    - Generate and save contingency tables
    """
    print(f"Starting Phase 5: Evaluation on {clustering_path}...")
    
    # 1. Load Data
    cluster_df = pd.read_csv(Path(clustering_path), index_col=0)
    meta_df = pd.read_csv(Path(metadata_path)).set_index("Sample Name")
    
    # Join to ensure alignment
    merged_df = cluster_df.join(meta_df[["Correct ID"]], how="inner")
    
    evaluator = Evaluator()
    results = []
    
    # 2. Evaluate K-Means
    print("Evaluating K-Means...")
    km_metrics = evaluator.calculate_metrics(merged_df["Correct ID"], merged_df["KMeans_Cluster"])
    km_metrics["Model"] = "K-Means"
    results.append(km_metrics)
    
    # 3. Evaluate Hierarchical
    print("Evaluating Hierarchical...")
    hier_metrics = evaluator.calculate_metrics(merged_df["Correct ID"], merged_df["Hierarchical_Cluster"])
    hier_metrics["Model"] = "Hierarchical"
    results.append(hier_metrics)
    
    # 4. Save Metrics
    metrics_df = pd.DataFrame(results).set_index("Model")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    metrics_df.to_csv(output_dir / "evaluation_metrics.csv")
    
    # 5. Generate and Save Contingency Tables
    km_table = evaluator.get_cluster_counts(merged_df, "KMeans_Cluster", "Correct ID")
    hier_table = evaluator.get_cluster_counts(merged_df, "Hierarchical_Cluster", "Correct ID")
    
    km_table.to_csv(output_dir / "contingency_kmeans.csv")
    hier_table.to_csv(output_dir / "contingency_hierarchical.csv")
    
    print("\nFinal Metrics:")
    print(metrics_df)
    print(f"\nResults saved to {output_dir}/")
    
    return metrics_df


if __name__ == "__main__":
    run_phase5()
