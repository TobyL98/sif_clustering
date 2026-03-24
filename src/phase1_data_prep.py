
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import DataLoader
from src.aligner import PeakAligner


def run_phase1(metadata_path: str = "sif_correct_scores.csv"):
    """
    Executes Phase 1 of the development plan:
    - Sample mapping
    - Peak alignment (Binning)
    - Feature construction
    
    The resulting feature matrix (X) is saved as a CSV where the index
    is 'Sample Name', making the link between features and metadata explicit.
    """
    print(f"Starting Phase 1 for {metadata_path}...")
    
    loader = DataLoader()
    aligner = PeakAligner() # Defaults to 0.4 Da and 800-3500 m/z
    
    # 1. Sample Mapping
    print("Mapping samples...")
    mapping_df = loader.get_sample_mapping(metadata_path)
    print(f"Total samples to process: {len(mapping_df)}")
    
    # 2 & 3. Binning and Feature Construction
    print("Aligning peaks and constructing feature matrix...")
    feature_list = []
    sample_names = []
    
    for _, row in mapping_df.iterrows():
        sample_peaks = loader.load_peaks(row["file_path"])
        feature_vector = aligner.align_sample(sample_peaks)
        feature_list.append(feature_vector)
        sample_names.append(row["Sample Name"])
        
    # Create feature dataframe (labeled with Sample Names and m/z bins)
    bin_centers = aligner.get_bin_centers()
    X_df = pd.DataFrame(
        np.stack(feature_list), 
        index=sample_names, 
        columns=[f"{mz:.2f}" for mz in bin_centers]
    )
    
    # 4. Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save feature matrix (labeled CSV is more robust than raw .npy)
    # We use float16 or float32 to save space if needed, but for now float64 is fine.
    X_df.to_csv(output_dir / "feature_matrix_phase1.csv")
    mapping_df.to_csv(output_dir / "sample_metadata_phase1.csv", index=False)
    
    print(f"Phase 1 complete. Feature matrix shape: {X_df.shape}")
    print(f"Results saved to {output_dir}/")
    return X_df, mapping_df


if __name__ == "__main__":
    run_phase1()
