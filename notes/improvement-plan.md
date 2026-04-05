# GIS-ML-Workflow Improvement Plan

This document captures what needs modernization given the project's age (~2020 era codebase running on Python 3.6+, sklearn <1.0, old Ray APIs).

---

## CRITICAL — Fix Before Public Release

### 1. Remove all `eval()` calls (25+ instances across 6 files)
**Risk:** Arbitrary code execution from config files.
**Where:** Every `pipeline.py` file uses `eval()` to parse model names, parameter dicts, feature lists, and scoring functions from `.ini` config.
**Fix:** Replace with a whitelist registry pattern:
```python
MODEL_REGISTRY = {
    "XGBRegressor": XGBRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    ...
}
model_class = MODEL_REGISTRY[config.get('Model', 'model_function')]
```
For parameter dicts, use `json.loads()` or `ast.literal_eval()` (safe subset of eval).
For scoring functions, use a similar registry.

### 2. Remove hardcoded Redis password (14 instances across 12 files)
**Current:** `ray.init(address='auto', _redis_password='5241590000000000', ...)`
**Fix:** Use environment variable: `os.environ.get('RAY_REDIS_PASSWORD', '')`

### 3. Fix unsafe `pickle.load()` (5 instances)
**Where:** `ensemble_models.py`, `cv_elimination.py`, `groupcv.py`, `feature_extraction/utlities.py`
**Fix:** Use `joblib.load()` (already used for SVM) consistently, or add file integrity checks.

### 4. Remove `from sklearn.datasets import load_boston`
**File:** `mlwkf/models/standard_models.py` line 7
**Reason:** Removed in scikit-learn 1.2 (ethical concerns). It's imported but never used in the class definitions. Just delete the import.

---

## HIGH — Deprecated APIs That Will Break

### 5. Fix deprecated sklearn `GradientBoostingRegressor` parameters
**File:** `mlwkf/models/ensemble_models.py`
- `loss='ls'` → `loss='squared_error'` (deprecated since sklearn 1.0)
- Remove `presort='deprecated'` parameter entirely (removed in sklearn 1.0)
- Remove `min_impurity_split` parameter (removed in sklearn 1.0)

### 6. Update Ray Tune API
**Current:** Uses `ray.tune.sample.uniform`, `tune.report()`, `tune.run()`
**Modern:** Ray 2.x uses `tune.Tuner`, `tune.TuneConfig`, `session.report()`, `ray.tune.search_space`
**Scope:** All 4 HPO algorithm files need updating.

### 7. Update XGBoost API
**Current:** Uses `xgb.Booster()` directly with `xgb.train()`
**Modern:** XGBoost 2.x deprecated `single_precision_histogram`, changed GPU params (`predictor: cpu_predictor` is gone).
**Scope:** `standard_models.py`, `bootstrapped_models.py`

### 8. Fix deprecated pandas APIs
- `df.append()` → `pd.concat()` (if used anywhere)
- `inplace=True` is discouraged in modern pandas; assign instead

---

## MEDIUM — Code Quality & Maintainability

### 9. Replace `print()` with `logging` (47+ instances)
Most pipeline files mix `logging.warning()` and `print()`. Standardize to logging throughout. Files:
- `prediction_mapping/pipeline.py`, `utlities.py`
- `covariates_drift/pipeline.py`, `utlities.py`
- `feature_extraction/algorithms/*.py`
- `model_exploration/pipeline.py`
- `bootstrapped_models.py`

### 10. Replace wildcard imports with explicit imports
All `from mlwkf.models.* import *` should list specific classes. Affects 20+ files.

### 11. Use context managers for file I/O
```python
# BAD (current)
pickle.dump(self.model, open(path, 'wb'))
# GOOD
with open(path, 'wb') as f:
    pickle.dump(self.model, f)
```

### 12. Add type hints to all public functions
Zero type hints exist. At minimum, add to:
- Model class interfaces (`fit`, `predict`, `save`, `load`)
- Utility functions
- Pipeline entry points

### 13. Remove blanket warning suppression
Multiple files do `warnings.simplefilter("ignore")`. Either fix the underlying warnings or use specific filters.

---

## MEDIUM — Dependency Cleanup

### 14. Trim `requirements.txt` from 159 → ~30 direct dependencies
**Current problem:** Lists transitive dependencies (e.g., `certifi`, `cffi`, `chardet`, `urllib3`). These should be resolved by pip automatically.
**Also remove:**
- `PySide6` — GUI library, not used by the pipeline
- `bcrypt` — not used
- `boto3`, `awscli` — AWS-specific, not core
- `jupyter*`, `notebook`, `nbconvert` — dev tools, not runtime deps
- `h2o` — imported but model class is skeletal/unused
- `chefboost` — only used by `RandomForestRegressor` which is a TODO stub
- `modin` — imported nowhere in actual pipeline code
- `torch` — only used for `torch.cuda.device_count()` in `utlities.py`; replace with a check on CUDA env vars or `shutil.which('nvidia-smi')`

### 15. Pin dependency versions
Currently no versions pinned. Add `>=` minimums for core deps:
```
xgboost>=1.7
scikit-learn>=1.2
ray[tune]>=2.0
rasterio>=1.3
fiona>=1.9
pandas>=1.5
numpy>=1.23
```

### 16. Migrate to `pyproject.toml` for all metadata
Currently split between `pyproject.toml` (build system only), `setup.cfg` (metadata), and `requirements.txt` (deps). Consolidate into a single `pyproject.toml` with `[project.dependencies]`.

---

## LOW — Architecture Improvements

### 17. Abstract model interface with Protocol/ABC
All models implement `fit/predict/save/load` but there's no formal interface. Add:
```python
class BaseModel(Protocol):
    def fit(self, data: pd.DataFrame, label: pd.Series, weight: pd.Series | None = None) -> None: ...
    def predict(self, data: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### 18. Complete the stub models
`LightGBMRegressor` and `RandomForestRegressor` are marked TODO and have broken interfaces (e.g., `fit(data)` instead of `fit(data, label)`). Either implement them properly or remove them from the supported models list.

### 19. Make config handling type-safe
Replace `configparser` + `eval()` with a structured config (Pydantic model or dataclass) that validates all fields at startup before any pipeline stage runs.

### 20. Add CI/CD pipeline
No CI configuration exists. Add GitHub Actions for:
- Lint (`ruff` or `flake8`)
- Type check (`mypy`)
- Unit tests (no GDAL required)
- Integration tests (with GDAL via Docker)

### 21. Separate data cleaning into a shared function
The pattern below is copy-pasted in 8+ files:
```python
df = df.astype('float32')
df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
```
`get_formated_dataframe()` in `utlities.py` does this, but most files duplicate it inline instead of calling the utility.

### 22. Make Ray optional for local development
Currently every pipeline stage hard-requires Ray. For small datasets and local dev, allow running without Ray by falling back to serial execution.

### 23. Add proper test data fixtures
Test GeoTIFF and GeoJSON files are not in the repo. Either:
- Generate synthetic fixtures in `conftest.py` using `rasterio` (small 10x10 GeoTIFFs)
- Or commit minimal test files to `tests/testdata/`

### 24. Fix typos in file/function names
- `utlities.py` → `utilities.py` (in every subpackage)
- `objective_funtions.py` → `objective_functions.py`
- Various docstring typos

---

## Summary Priority Matrix

| Priority | Items | Effort |
|----------|-------|--------|
| CRITICAL | #1-4 (eval, credentials, pickle, load_boston) | 1-2 days |
| HIGH | #5-8 (deprecated APIs) | 2-3 days |
| MEDIUM | #9-16 (code quality, deps) | 3-5 days |
| LOW | #17-24 (architecture) | 1-2 weeks |

**Recommended order:** Fix CRITICAL items first (security), then HIGH (they'll cause import errors with modern library versions), then tackle MEDIUM and LOW as part of a modernization sprint.
