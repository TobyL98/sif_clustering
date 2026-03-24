
import pandas as pd
from pathlib import Path
from src.evaluation import Evaluator


def run_phase5(
    clustering_path: str = "results/clustering_results.csv",
    metadata_path: str = "results/sample_metadata_phase1.csv"
):
    """
    Executes Phase 5: Comprehensive evaluation of all clustering models.
    """
    print(f"Starting Phase 5: Extended Evaluation on {clustering_path}...")
    
    cluster_df = pd.read_csv(Path(clustering_path), index_col=0)
    meta_df = pd.read_csv(Path(metadata_path)).set_index("Sample Name")
    merged_df = cluster_df.join(meta_df[["Correct ID"]], how="inner")
    
    evaluator = Evaluator()
    results = []
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Iterate through each clustering method
    for col in cluster_df.columns:
        print(f"Evaluating {col}...")
        metrics = evaluator.calculate_metrics(merged_df["Correct ID"], merged_df[col])
        metrics["Model"] = col
        results.append(metrics)
        
        # Save individual contingency table
        table = evaluator.get_cluster_counts(merged_df, col, "Correct ID")
        table.to_csv(output_dir / f"contingency_{col}.csv")
    
    metrics_df = pd.DataFrame(results).set_index("Model")
    metrics_df.to_csv(output_dir / "evaluation_metrics.csv")
    
    print("\nFinal Metrics Comparison:")
    print(metrics_df.sort_values("V-Measure", ascending=False))
    
    return metrics_df


if __name__ == "__main__":
    run_phase5()
