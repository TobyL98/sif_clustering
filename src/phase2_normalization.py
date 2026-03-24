
import pandas as pd
from pathlib import Path
from src.normalization import Normalizer


def run_phase2(input_path: str = "results/feature_matrix_phase1.csv"):
    """
    Executes Phase 2 of the development plan:
    - TIC Normalization
    - Log Transformation
    - Binary Encoding
    """
    print(f"Starting Phase 2: Normalization & Transformation on {input_path}...")
    
    # Check if input file exists
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input feature matrix not found at {input_path}. Please run Phase 1 first.")
    
    # 1. Load Phase 1 data
    X_df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded matrix of shape: {X_df.shape}")
    
    # 2. TIC Normalization (Baseline)
    print("Applying TIC normalization...")
    X_tic = Normalizer.tic_normalize(X_df)
    
    # 3. Log Transformation (on TIC data)
    print("Applying log transformation...")
    X_log = Normalizer.log_transform(X_tic)
    
    # 4. Binary Encoding (on raw data)
    print("Applying binary transformation...")
    X_binary = Normalizer.binary_transform(X_df)
    
    # 5. Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    X_tic.to_csv(output_dir / "feature_matrix_tic.csv")
    X_log.to_csv(output_dir / "feature_matrix_tic_log.csv")
    X_binary.to_csv(output_dir / "feature_matrix_binary.csv")
    
    print(f"Phase 2 complete. Generated 3 normalized versions in {output_dir}/")
    return X_binary # Prioritizing binary for next phase visualization


if __name__ == "__main__":
    run_phase2()
