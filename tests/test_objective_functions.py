import numpy as np
import pytest
import xgboost as xgb
from mlwkf.objective_functions import gradient, hessian, squared_log


@pytest.fixture
def dtrain():
    """Create a small DMatrix for testing."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    return xgb.DMatrix(X, label=y)


class TestGradient:

    def test_zero_gradient_at_perfect_prediction(self, dtrain):
        y = dtrain.get_label()
        grad = gradient(y, dtrain)
        np.testing.assert_array_almost_equal(grad, np.zeros_like(y), decimal=5)

    def test_returns_array_same_shape(self, dtrain):
        predt = np.array([1.5, 2.5, 3.5])
        grad = gradient(predt, dtrain)
        assert grad.shape == predt.shape


class TestHessian:

    def test_returns_array_same_shape(self, dtrain):
        predt = np.array([1.5, 2.5, 3.5])
        hess = hessian(predt, dtrain)
        assert hess.shape == predt.shape


class TestSquaredLog:

    def test_returns_grad_and_hess(self, dtrain):
        predt = np.array([1.5, 2.5, 3.5])
        grad, hess = squared_log(predt, dtrain)
        assert grad.shape == predt.shape
        assert hess.shape == predt.shape

    def test_clamps_negative_predictions(self, dtrain):
        predt = np.array([-5.0, -2.0, 3.0])
        grad, hess = squared_log(predt, dtrain)
        # Should not raise; clamped values handled
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
