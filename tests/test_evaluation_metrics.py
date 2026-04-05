import numpy as np
import pytest
from mlwkf.evaluation_metrics import (
    mean_squared_error_scorer,
    mean_absolute_error_scorer,
    r2_scorer,
    rmse_scorer,
    adjusted_r2_scorer,
)


class TestMeanSquaredErrorScorer:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error_scorer(y, y) == 0.0

    def test_returns_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        score = mean_squared_error_scorer(y_true, y_pred)
        assert score < 0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # MSE = (1+1+1)/3 = 1.0, negated = -1.0
        assert mean_squared_error_scorer(y_true, y_pred) == pytest.approx(-1.0)


class TestMeanAbsoluteErrorScorer:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error_scorer(y, y) == 0.0

    def test_returns_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        score = mean_absolute_error_scorer(y_true, y_pred)
        assert score < 0

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 3.0])
        # MAE = (1+3)/2 = 2.0, negated = -2.0
        assert mean_absolute_error_scorer(y_true, y_pred) == pytest.approx(-2.0)


class TestR2Scorer:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_scorer(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # predicting mean
        assert r2_scorer(y_true, y_pred) == pytest.approx(0.0)

    def test_bad_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert r2_scorer(y_true, y_pred) < 0


class TestRmseScorer:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse_scorer(y, y) == 0.0

    def test_returns_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert rmse_scorer(y_true, y_pred) < 0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # RMSE = sqrt(1.0) = 1.0, negated = -1.0
        assert rmse_scorer(y_true, y_pred) == pytest.approx(-1.0)


class TestAdjustedR2Scorer:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = adjusted_r2_scorer(y, y, 2)
        assert score == pytest.approx(1.0)

    def test_more_covariates_penalizes(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1])
        score_few = adjusted_r2_scorer(y_true, y_pred, 1)
        score_many = adjusted_r2_scorer(y_true, y_pred, 5)
        assert score_few >= score_many

    def test_floors_at_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        score = adjusted_r2_scorer(y_true, y_pred, 1)
        assert score == 0.0
