import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from mlwkf.models.standard_models import XGBRegressor, CatBoostRegressor, SVMRegressor
from mlwkf.models.bootstrapped_models import BootstrappedXGBRegressor, BootstrappedSVMRegressor
from mlwkf.models.ensemble_models import SuperLearnerRegressor


@pytest.fixture
def training_data():
    """Synthetic regression dataset."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
    })
    y = 2 * X["feat_1"] + 0.5 * X["feat_2"] + np.random.randn(n) * 0.1
    return X, y


@pytest.fixture
def prediction_data():
    """Synthetic prediction dataset."""
    np.random.seed(99)
    n = 20
    return pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
    })


class TestXGBRegressor:

    def test_default_init(self):
        model = XGBRegressor()
        assert model.param["booster"] == "gbtree"
        assert model.num_boost_round == 500

    def test_custom_params(self):
        model = XGBRegressor({"max_depth": 5, "num_boost_round": 50})
        assert model.param["max_depth"] == 5
        assert model.num_boost_round == 50

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = XGBRegressor({"num_boost_round": 10, "max_depth": 3})
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)
        assert not np.any(np.isnan(preds))

    def test_save_load(self, training_data, prediction_data, tmp_path):
        X_train, y_train = training_data
        model = XGBRegressor({"num_boost_round": 10, "max_depth": 3})
        model.fit(X_train, y_train)
        preds_before = model.predict(prediction_data)

        model_path = str(tmp_path / "model.bin")
        model.save(model_path)

        model2 = XGBRegressor({"num_boost_round": 10, "max_depth": 3})
        model2.load(model_path)
        preds_after = model2.predict(prediction_data)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestSVMRegressor:

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = SVMRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)
        assert not np.any(np.isnan(preds))

    def test_save_load(self, training_data, prediction_data, tmp_path):
        X_train, y_train = training_data
        model = SVMRegressor()
        model.fit(X_train, y_train)
        preds_before = model.predict(prediction_data)

        model_path = str(tmp_path / "model.joblib")
        model.save(model_path)

        model2 = SVMRegressor()
        model2.load(model_path)
        preds_after = model2.predict(prediction_data)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestCatBoostRegressor:

    def test_default_init(self):
        model = CatBoostRegressor()
        assert model.param["depth"] == 15

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = CatBoostRegressor({"iterations": 10, "depth": 3, "verbose": 0})
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)
        assert not np.any(np.isnan(preds))


class TestBootstrappedXGBRegressor:

    def test_default_init(self):
        model = BootstrappedXGBRegressor()
        assert model.bootstrapped_number_of_models == 10

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = BootstrappedXGBRegressor({
            "num_boost_round": 5,
            "max_depth": 3,
            "bootstrapped_number_of_models": 3,
        })
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)
        assert not np.any(np.isnan(preds))

    def test_stores_individual_predictions(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = BootstrappedXGBRegressor({
            "num_boost_round": 5,
            "max_depth": 3,
            "bootstrapped_number_of_models": 3,
        })
        model.fit(X_train, y_train)
        model.predict(prediction_data)
        assert model.predicted_results.shape == (3, len(prediction_data))


class TestBootstrappedSVMRegressor:

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = BootstrappedSVMRegressor({
            "bootstrapped_number_of_models": 2,
        })
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)


class TestSuperLearnerRegressor:

    def test_fit_predict(self, training_data, prediction_data):
        X_train, y_train = training_data
        model = SuperLearnerRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(prediction_data)
        assert len(preds) == len(prediction_data)
        assert not np.any(np.isnan(preds))

    def test_save_load(self, training_data, prediction_data, tmp_path):
        X_train, y_train = training_data
        model = SuperLearnerRegressor()
        model.fit(X_train, y_train)
        preds_before = model.predict(prediction_data)

        model_path = str(tmp_path / "model.pkl")
        model.save(model_path)

        model2 = SuperLearnerRegressor()
        model2.load(model_path)
        preds_after = model2.predict(prediction_data)

        np.testing.assert_array_almost_equal(preds_before, preds_after)
