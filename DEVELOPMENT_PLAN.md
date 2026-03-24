# ZooMS Clustering Development Plan

## 1. Objective
Perform unsupervised clustering on ZooMS (Zoo Archaeology by Mass Spectrometry) peak lists to group samples into their correct taxonomic families, validating the results against manual annotations.

## 2. Data Assets
- **Peak Lists**: Aligned text files in `peaklist_inputs/PinHoleDataset_SIMPER/<Family>/peak_list/` (Format: m/z \t intensity).
- **Ground Truth**: `sif_correct_scores.csv` (Maps Sample Name to Correct ID).

## 3. Workflow Phases

### Phase 1: Data Preparation & Alignment
- **Sample Mapping**: Create a registry linking the Sample Name in the CSV to the corresponding .txt file path across family subdirectories.
- **Peak Alignment (Binning)**: Since $m/z$ values vary slightly between runs, implement a binning strategy (e.g., rounding to 1.0 Da or using a kernel-based alignment) to create a consistent feature matrix.
- **Feature Construction**: Build a matrix where rows are samples and columns are $m/z$ bins.

### Phase 2: Normalization & Transformation
- **Scaling**: Apply Total Ion Current (TIC) normalization to account for differences in overall sample concentration.
- **Variance Stabilization**: Test log-transformation or binary encoding (presence/absence) to prevent high-intensity peaks from overwhelming the signal.

### Phase 3: Dimensionality Reduction
- **PCA/t-SNE**: Reduce the feature space to visualize variance and assess if families naturally separate before applying clustering algorithms.

### Phase 4: Clustering Implementation
- **Hierarchical Clustering**: Use Ward’s linkage (Agglomerative) to mirror potential biological/taxonomic hierarchies.
- **K-Means**: Baseline centroid-based clustering.
- **DBSCAN**: Test for density-based grouping to identify noise or outliers.

### Phase 5: Evaluation & Validation
- **Metric Calculation**: Use Adjusted Rand Index (ARI) and V-Measure to compare cluster assignments with the Correct ID from the ground truth.
- **Visualization**: Generate dendrograms and scatter plots (PCA/UMAP) colored by both cluster ID and ground truth label.
