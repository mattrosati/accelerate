"""Shared patient grouping and weighting utilities.

These functions are used by both the classical ML and deep learning pipelines
to ensure consistent patient-level evaluation and balancing.
"""

import numpy as np


def infer_base_ptid(labels):
    """Return the patient identifier used for grouped splitting/evaluation."""
    if "base_ptid" in labels.columns:
        return labels["base_ptid"].astype(str).to_numpy()
    return labels["ptid"].astype(str).str.split("_").str[0].to_numpy()


def make_patient_weights(groups):
    """Assign each window inverse-frequency weight within its patient."""
    groups = np.asarray(groups).astype(str)
    if groups.size == 0:
        return np.array([], dtype=np.float64)
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_map = dict(zip(unique_groups, counts))
    weights = np.array([1.0 / count_map[g] for g in groups], dtype=np.float64)
    return weights * (len(weights) / weights.sum())
