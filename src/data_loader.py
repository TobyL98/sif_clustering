
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class DataLoader:
    """
    Handles loading and mapping of ZooMS peak lists.
    """

    def __init__(self, root_dir: str = "."):
        """
        Initialize the DataLoader.

        Args:
            root_dir (str): Project root directory.
        """
        self.root_dir = Path(root_dir)
        # Search in the standard input directory
        self.peaklist_dir = self.root_dir / "peaklist_inputs"

    def get_sample_mapping(self, metadata_path: str) -> pd.DataFrame:
        """
        Create a mapping between sample names in a CSV and their file paths.

        Args:
            metadata_path (str): Path to the CSV containing 'Sample Name'.

        Returns:
            pd.DataFrame: DataFrame containing metadata and 'file_path'.
        """
        df = pd.read_csv(self.root_dir / metadata_path)
        
        # Create a dictionary of all available peaklist files for fast lookup
        # key: sample_name, value: path
        available_files = {}
        for txt_path in self.peaklist_dir.rglob("*.txt"):
            # Filenames are expected to be <sample_name>_peaklist.txt
            # We strip the suffix to get the sample name
            sample_name = txt_path.name.replace("_peaklist.txt", "")
            # Also handle the case where the extension might just be .txt (selected/ folder)
            if sample_name.endswith(".txt"):
                 sample_name = sample_name[:-4]
            
            available_files[sample_name] = txt_path

        # Map paths to the dataframe using 'Sample Name' as the key
        df["file_path"] = df["Sample Name"].map(lambda x: str(available_files.get(x)) if available_files.get(x) else None)
        
        # Check for missing files
        missing = df[df["file_path"].isna()]
        if not missing.empty:
            print(f"Warning: {len(missing)} samples in CSV have no matching .txt file.")
            
        return df.dropna(subset=["file_path"]).copy()

    def load_peaks(self, file_path: str) -> pd.DataFrame:
        """
        Load m/z and intensity values from a peaklist file.

        Args:
            file_path (str): Path to the .txt peaklist file.

        Returns:
            pd.DataFrame: DataFrame with 'mz' and 'intensity' columns.
        """
        return pd.read_csv(
            file_path, 
            sep="\t", 
            header=None, 
            names=["mz", "intensity"]
        )
