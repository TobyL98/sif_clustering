
import pytest
import pandas as pd
import numpy as np
from src.evaluation import Evaluator

def test_calculate_metrics_perfect():
    """Verify that perfect clustering gives scores of 1.0."""
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    
    metrics = Evaluator.calculate_metrics(truth, pred)
    
    assert metrics["ARI"] == 1.0
    assert metrics["V-Measure"] == 1.0
    assert metrics["Homogeneity"] == 1.0
    assert metrics["Completeness"] == 1.0

def test_calculate_metrics_shuffled():
    """Verify that scores are 1.0 even if labels are named differently."""
    truth = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0]) # Labels are flipped but grouping is the same
    
    metrics = Evaluator.calculate_metrics(truth, pred)
    
    assert metrics["ARI"] == 1.0
    assert metrics["V-Measure"] == 1.0

def test_get_cluster_counts():
    """Verify that the contingency table is created correctly."""
    df = pd.DataFrame({
        "Cluster": [0, 0, 1, 1],
        "Truth": ["A", "A", "B", "C"]
    })
    
    table = Evaluator.get_cluster_counts(df, "Cluster", "Truth")
    
    # Cluster 0 should have 2 from family A
    assert table.loc[0, "A"] == 2
    # Cluster 1 should have 1 from family B and 1 from C
    assert table.loc[1, "B"] == 1
    assert table.loc[1, "C"] == 1
