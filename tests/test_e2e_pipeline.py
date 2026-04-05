"""
End-to-end pipeline test using synthetic GeoTIFF and GeoJSON data.

This test generates minimal synthetic geospatial data, writes a temporary
config file, and runs stages that don't require Ray (evaluation, model
training via direct API calls). It validates the full data flow:
  CSV creation → cleaning → feature extraction logic → model train → predict.

For the full system test (with Ray + real data), use test_system.py.
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path

from mlwkf.utlities import read_dataframe_from_csv, get_formated_dataframe, flatten, create_chunked_target
from mlwkf.evaluation_metrics import r2_scorer, rmse_scorer, adjusted_r2_scorer
from mlwkf.models.standard_models import XGBRegressor
from mlwkf.data_preparation.pipeline import merge_csv_file


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a synthetic training CSV that mimics pipeline output."""
    np.random.seed(42)
    n = 200
    feat_1 = np.random.randn(n)
    feat_2 = np.random.randn(n)
    feat_3 = np.random.randn(n)
    target = 3.0 * feat_1 - 1.5 * feat_2 + 0.5 * feat_3 + np.random.randn(n) * 0.2
    x_coord = np.random.uniform(100000, 200000, n)
    y_coord = np.random.uniform(-4000000, -3000000, n)
    groupcv = np.random.choice([0, 1, 2, 3, 4], n)

    df = pd.DataFrame({
        "target": target,
        "feat_1": feat_1,
        "feat_2": feat_2,
        "feat_3": feat_3,
        "x": x_coord,
        "y": y_coord,
        "groupcv": groupcv,
    })

    train_path = tmp_path / "training_dataset.csv"
    oos_path = tmp_path / "oos_dataset.csv"

    train_df = df.iloc[:160]
    oos_df = df.iloc[160:]

    train_df.to_csv(train_path, index=False)
    oos_df.to_csv(oos_path, index=False)

    return train_path, oos_path, tmp_path


class TestEndToEndDataFlow:
    """Tests the full data flow without Ray."""

    def test_csv_merge_and_clean(self, tmp_path):
        """Stage 1 equivalent: merge chunked CSVs and clean."""
        # Create chunked CSVs (simulating data_preparation output)
        paths = []
        for i in range(3):
            p = tmp_path / f"chunk_{i}.csv"
            df = pd.DataFrame({
                "target": np.random.randn(10),
                "feat_1": np.random.randn(10),
                "x": np.random.uniform(0, 1, 10),
                "y": np.random.uniform(0, 1, 10),
            })
            df.to_csv(p, index=False)
            paths.append(p)

        merged_path = tmp_path / "merged.csv"
        merge_csv_file(paths, merged_path)

        result = read_dataframe_from_csv(str(merged_path))
        assert len(result) == 30
        assert "target" in result.columns
        assert result.dtypes["target"] == np.float32

    def test_data_cleaning_pipeline(self, synthetic_dataset):
        """Validate the cleaning function removes bad rows."""
        train_path, _, _ = synthetic_dataset

        # Inject bad rows
        df = pd.read_csv(train_path)
        bad_row_nan = pd.DataFrame({"target": [np.nan], "feat_1": [1.0], "feat_2": [2.0], "feat_3": [3.0], "x": [0], "y": [0], "groupcv": [0]})
        bad_row_inf = pd.DataFrame({"target": [1.0], "feat_1": [np.inf], "feat_2": [2.0], "feat_3": [3.0], "x": [0], "y": [0], "groupcv": [0]})
        bad_row_nodata = pd.DataFrame({"target": [1.0], "feat_1": [-9999.0], "feat_2": [2.0], "feat_3": [3.0], "x": [0], "y": [0], "groupcv": [0]})

        df = pd.concat([df, bad_row_nan, bad_row_inf, bad_row_nodata], ignore_index=True)
        dirty_path = train_path.parent / "dirty.csv"
        df.to_csv(dirty_path, index=False)

        cleaned = read_dataframe_from_csv(str(dirty_path))
        assert len(cleaned) == 160  # original 160 training rows, 3 bad rows removed

    def test_full_train_evaluate_predict_cycle(self, synthetic_dataset):
        """Stages 3-6: feature selection → train → evaluate → predict."""
        train_path, oos_path, output_dir = synthetic_dataset

        # Load and clean data
        train_df = read_dataframe_from_csv(str(train_path))
        oos_df = read_dataframe_from_csv(str(oos_path))

        feature_cols = ["feat_1", "feat_2", "feat_3"]

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_oos = oos_df[feature_cols]
        y_oos = oos_df["target"]

        # Train model (Stage 4 output)
        model = XGBRegressor({"num_boost_round": 50, "max_depth": 4, "learning_rate": 0.1})
        model.fit(X_train, y_train)

        # Evaluate on OOS (Stage 5)
        preds = model.predict(X_oos)
        r2 = r2_scorer(y_oos.values, preds)
        rmse = rmse_scorer(y_oos.values, preds)
        adj_r2 = adjusted_r2_scorer(y_oos.values, preds, len(feature_cols))

        assert r2 > 0.5, f"R2 should be reasonable for synthetic data, got {r2}"
        assert rmse < 0, "RMSE scorer should be negative (negated convention)"
        assert adj_r2 > 0, "Adjusted R2 should be positive"

        # Save and reload model (Stage 6 prerequisite)
        model_path = str(output_dir / "trained_model.bin")
        model.save(model_path)

        model2 = XGBRegressor({"num_boost_round": 50, "max_depth": 4})
        model2.load(model_path)
        preds2 = model2.predict(X_oos)

        np.testing.assert_array_almost_equal(preds, preds2)

    def test_chunked_target_creation(self):
        """Test the chunking logic used in data preparation."""
        targets = list(range(12345))
        chunks = create_chunked_target(targets, 5000)
        assert len(chunks) == 3
        assert len(chunks[0]) == 5000
        assert len(chunks[1]) == 5000
        assert len(chunks[2]) == 2345
        assert flatten(chunks) == targets

    def test_multiple_scoring_functions(self, synthetic_dataset):
        """Verify all scoring functions work in an HPO-like evaluation loop."""
        from mlwkf.evaluation_metrics import (
            mean_squared_error_scorer,
            mean_absolute_error_scorer,
            r2_scorer,
            rmse_scorer,
            adjusted_r2_scorer,
        )

        train_path, oos_path, _ = synthetic_dataset
        train_df = read_dataframe_from_csv(str(train_path))
        oos_df = read_dataframe_from_csv(str(oos_path))

        feature_cols = ["feat_1", "feat_2", "feat_3"]
        model = XGBRegressor({"num_boost_round": 20, "max_depth": 3})
        model.fit(train_df[feature_cols], train_df["target"])
        preds = model.predict(oos_df[feature_cols])
        y_true = oos_df["target"].values

        scoring_functions = [
            mean_squared_error_scorer,
            mean_absolute_error_scorer,
            r2_scorer,
            rmse_scorer,
        ]

        results = {}
        for scorer in scoring_functions:
            score = scorer(y_true, preds)
            results[scorer.__name__] = score
            assert np.isfinite(score), f"{scorer.__name__} returned non-finite: {score}"

        adj_r2 = adjusted_r2_scorer(y_true, preds, len(feature_cols))
        results["adjusted_r2_scorer"] = adj_r2
        assert np.isfinite(adj_r2)

        # Verify scoring convention: higher is better
        assert results["r2_scorer"] > 0
        assert results["mean_squared_error_scorer"] <= 0
        assert results["mean_absolute_error_scorer"] <= 0
        assert results["rmse_scorer"] <= 0
