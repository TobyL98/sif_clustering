# ZooMS Clustering Development Plan

## 1. Objective
Perform unsupervised clustering on ZooMS (Zoo Archaeology by Mass Spectrometry) peak lists to group samples into their correct taxonomic families, comparing binary presence/absence against intensity-weighted feature sets.

## 2. Data Assets
*   **Peak Lists**: Aligned text files in `peaklist_inputs/PinHoleDataset_SIMPER/<Family>/peak_list/` (Format: `m/z` \t `intensity`).
*   **Ground Truth**: `sif_correct_scores.csv` (Maps `Sample Name` to `Correct ID`).

## 3. Workflow Phases

### Phase 1: Data Preparation & Alignment
*   **Sample Mapping**: Build a registry linking the `Sample Name` in the CSV to the actual `.txt` file paths. Identify any missing files or mismatched names.
*   **Peak Alignment (Binning)**: Implement a binning strategy (e.g., 1.0 Da or 0.5 Da resolution) to align sparse $m/z$ values across all samples.
*   **Base Matrix Construction**: Create a master coordinate-format matrix where each row is a sample and each column is a binned $m/z$ value.

### Phase 2: Feature Engineering (Dual Strategy)
*   **Strategy A: Binary Encoding (Presence/Absence)**
    *   Convert all non-zero intensities to `1`.
    *   Focuses strictly on the existence of diagnostic "marker peptides" regardless of their abundance.
*   **Strategy B: Intensity-Weighted Encoding (TIC Normalized)**
    *   Apply Total Ion Current (TIC) normalization (divide each peak by the sum of all peaks in that sample).
    *   Accounts for overall sample concentration while preserving the relative height of peaks.
*   **Comparison Preparation**: Generate two distinct feature matrices to evaluate which better captures taxonomic variance.

### Phase 3: Dimensionality Reduction & Visualization
*   **PCA / t-SNE / UMAP**: Reduce the feature space to 2D/3D.
*   **Visual Audit**: Color points by "Correct ID" (from CSV) to see if families naturally cluster better under Strategy A or Strategy B.

### Phase 4: Clustering Implementation
*   **Hierarchical Clustering**: Use Ward’s linkage to build a dendrogram (useful for seeing taxonomic hierarchies).
*   **K-Means**: Baseline centroid-based clustering.
*   **DBSCAN/HDBSCAN**: Identify dense clusters and separate them from "noise" or low-quality samples.

### Phase 5: Evaluation & Validation
*   **Performance Metrics**: Calculate Adjusted Rand Index (ARI), V-Measure, and Silhouette Scores for both strategies.
*   **Error Analysis**: Specifically investigate samples like `20140121_PH48SOLrun_0_H3` (Rhinocerotidae) to see if unsupervised clustering correctly groups them with their family or repeats previous identification errors.

## Code Style

Follow PEP8 and Google style guide with the following additional rules:

### Docstrings
Napoleon (Google) style, opening and closing quotes on their own lines. Use type hints instead of types in docstrings.

## Markdown standards

- Always run markdownlint on any markdown files created or edited
- Install using: `pixi global install markdownlint-cli`
- Fix all linting issues before completing the task

## Testing preferences

- Write all Python tests as `pytest` style functions, not unittest classes
- Use descriptive function names starting with `test_`
- Prefer fixtures over setup/teardown methods
- Use assert statements directly, not self.assertEqual

## Testing approach

- Never create throwaway test scripts or ad hoc verification files
- If you need to test functionality, write a proper test in the test suite
- All tests go in the `tests/` directory following the project structure
- Tests should be runnable with the rest of the suite (`pixi run pytest`)
- Even for quick verification, write it as a real test that provides ongoing value

## Package management

- This project uses Pixi for all package management
- Never run commands directly (python, pytest, etc.)
- Always prefix commands with `pixi run <command>`
- Example: `pixi run python script.py` not `python script.py`
- Example: `pixi run pytest` not `pytest`
