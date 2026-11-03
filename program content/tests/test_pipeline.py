"""Smoke tests - run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np

from generate_dataset import generate
from segmentation import DEFAULT_FEATURES, evaluate_k_range, fit_final_model, preprocess


def test_dataset_shape_and_ranges():
    df = generate(300, seed=1)
    assert len(df) == 300
    assert df["Spending_Score"].between(1, 100).all()
    assert df["Age"].between(18, 70).all()
    assert not df.isna().any().any()


def test_scaling_is_standardized():
    X, _ = preprocess(generate(300, seed=1), DEFAULT_FEATURES)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X.std(axis=0), 1, atol=1e-9)


def test_kmeans_is_deterministic_and_finds_five_groups():
    X, _ = preprocess(generate(500, seed=42), DEFAULT_FEATURES)
    metrics = evaluate_k_range(X, 2, 8)
    assert int(metrics.loc[metrics["silhouette"].idxmax(), "k"]) == 5
    a, b = fit_final_model(X, 5).labels_, fit_final_model(X, 5).labels_
    assert (a == b).all()
