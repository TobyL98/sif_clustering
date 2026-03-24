
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.dim_reduction import DimensionalityReducer


def run_phase3(
    feature_path: str = "results/feature_matrix_binary.csv",
    metadata_path: str = "results/sample_metadata_phase1.csv"
):
    """
    Executes Phase 3 of the development plan:
    - Load feature matrix and metadata
    - Perform PCA and t-SNE
    - Generate visualizations colored by Correct ID
    """
    print(f"Starting Phase 3: Dimensionality Reduction & Visualization on {feature_path}...")
    
    # 1. Load Data
    X_df = pd.read_csv(Path(feature_path), index_col=0)
    meta_df = pd.read_csv(Path(metadata_path))
    
    # Ensure indices match
    # We join meta_df (which has Sample Name) with X_df index
    meta_df = meta_df.set_index("Sample Name").loc[X_df.index]
    
    reducer = DimensionalityReducer()
    
    # 2. Run PCA
    print("Running PCA...")
    pca_df = reducer.run_pca(X_df)
    
    # 3. Run t-SNE
    print("Running t-SNE...")
    tsne_df = reducer.run_tsne(X_df)
    
    # 4. Visualize
    print("Generating plots...")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA Plot
    sns.scatterplot(
        x=pca_df.iloc[:, 0], 
        y=pca_df.iloc[:, 1], 
        hue=meta_df["Correct ID"],
        ax=axes[0],
        palette="tab10",
        alpha=0.7
    )
    axes[0].set_title("PCA: Binary Feature Matrix")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # t-SNE Plot
    sns.scatterplot(
        x=tsne_df.iloc[:, 0], 
        y=tsne_df.iloc[:, 1], 
        hue=meta_df["Correct ID"],
        ax=axes[1],
        palette="tab10",
        alpha=0.7
    )
    axes[1].set_title("t-SNE: Binary Feature Matrix")
    axes[1].get_legend().remove() # Use one legend for both
    
    plt.tight_layout()
    plot_file = output_dir / "dim_reduction_binary.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Phase 3 complete. Plots saved to {plot_file}")
    return pca_df, tsne_df


if __name__ == "__main__":
    run_phase3()
