"""
Safe registries for dynamic lookup of models, scoring functions, feature
extraction algorithms, HPO algorithms, and binary classifiers.

Replaces all eval() usage across the codebase with whitelist-based lookups.
"""
import ast
import os
import logging

from mlwkf.models.standard_models import (
    XGBRegressor,
    CatBoostRegressor,
    LightGBMRegressor,
    RandomForestRegressor,
    SVMRegressor,
)
from mlwkf.models.bootstrapped_models import (
    BootstrappedXGBRegressor,
    BootstrappedSVMRegressor,
)
from mlwkf.models.ensemble_models import (
    SuperLearnerRegressor,
    QuantileGradientBoostingRegressor,
)
from mlwkf.evaluation_metrics import (
    mean_squared_error_scorer,
    mean_absolute_error_scorer,
    r2_scorer,
    rmse_scorer,
    adjusted_r2_scorer,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


MODEL_REGISTRY = {
    "XGBRegressor": XGBRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    "LightGBMRegressor": LightGBMRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "SVMRegressor": SVMRegressor,
    "BootstrappedXGBRegressor": BootstrappedXGBRegressor,
    "BootstrappedSVMRegressor": BootstrappedSVMRegressor,
    "SuperLearnerRegressor": SuperLearnerRegressor,
    "QuantileGradientBoostingRegressor": QuantileGradientBoostingRegressor,
}

SCORING_FUNCTION_REGISTRY = {
    "mean_squared_error_scorer": mean_squared_error_scorer,
    "mean_absolute_error_scorer": mean_absolute_error_scorer,
    "r2_scorer": r2_scorer,
    "rmse_scorer": rmse_scorer,
    "adjusted_r2_scorer": adjusted_r2_scorer,
}

BINARY_CLASSIFIER_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GaussianNB": GaussianNB,
    "KNeighborsClassifier": KNeighborsClassifier,
}


def lookup_model(name: str):
    """Look up a model class by name from the whitelist registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def lookup_scoring_function(name: str):
    """Look up a scoring function by name from the whitelist registry."""
    if name not in SCORING_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown scoring function '{name}'. "
            f"Supported: {list(SCORING_FUNCTION_REGISTRY.keys())}"
        )
    return SCORING_FUNCTION_REGISTRY[name]


def lookup_scoring_functions(names_str: str):
    """Parse a config string like '[r2_scorer, rmse_scorer]' into a list of functions."""
    # Strip brackets and whitespace, split by comma
    cleaned = names_str.strip().strip("[]")
    names = [n.strip() for n in cleaned.split(",") if n.strip()]
    return [lookup_scoring_function(name) for name in names]


def lookup_binary_classifier(name: str):
    """Look up a binary classifier class by name from the whitelist registry."""
    if name not in BINARY_CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Supported: {list(BINARY_CLASSIFIER_REGISTRY.keys())}"
        )
    return BINARY_CLASSIFIER_REGISTRY[name]


def safe_parse_dict(value: str) -> dict:
    """Safely parse a dict string from config using ast.literal_eval.

    For hyperparameter dicts that contain Ray Tune sampling functions
    (uniform, loguniform, etc.), those must be parsed separately since
    ast.literal_eval only handles literals.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback: empty dict if parsing fails
        logging.warning(f"Could not safely parse dict: {value!r}, returning empty dict")
        return {}


def safe_parse_list(value: str) -> list:
    """Safely parse a list string from config using ast.literal_eval."""
    if value is None or value == "None":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        logging.warning(f"Could not safely parse list: {value!r}, returning empty list")
        return []


def get_ray_redis_password() -> str:
    """Get Ray Redis password from environment variable."""
    return os.environ.get("RAY_REDIS_PASSWORD", "5241590000000000")
