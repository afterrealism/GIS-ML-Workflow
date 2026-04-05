# GIS-ML-Workflow

Machine learning workflow automation for geoscience. Processes GeoTIFF raster data through a configurable 7-stage pipeline — from data preparation to spatial prediction mapping.

## Features

- Preprocess GeoTIFF covariates and vector targets into ML-ready datasets
- Feature ranking via SHAP, recursive elimination (CV/OOS), group CV, or random baseline
- Hyperparameter tuning with Bayesian optimization, HyperOpt (TPE), or grid search
- Model exploration with k-fold cross-validation and out-of-sample evaluation
- Spatial prediction mapping to GeoTIFF output
- Covariate drift detection between training and prediction domains
- Distributed execution via Ray

## Supported Models

| Type | Models |
|------|--------|
| Standard | XGBRegressor, CatBoostRegressor, SVMRegressor |
| Ensemble | SuperLearnerRegressor, QuantileGradientBoostingRegressor |
| Bootstrapped | BootstrappedXGBRegressor, BootstrappedSVMRegressor |

## Pipeline Stages

Each stage is toggled on/off in the `.ini` config file:

1. **DataPreparation** — Reprojects rasters to EPSG:3577, extracts covariate values at target locations, creates training/OOS CSVs
2. **DataExploration** — Generates distribution plots, scatter plots, and geospatial heatmaps
3. **FeatureExtraction** — Ranks and selects features using pluggable algorithms
4. **HyperParameterOptimization** — Tunes model hyperparameters via Ray Tune
5. **ModelExploration** — Validates trained model with CV and OOS testing
6. **PredictionMapping** — Creates spatial prediction GeoTIFF from trained model
7. **CovariateDrift** — Detects distribution shift between training and prediction data

## Installation

### Requirements

- Python >= 3.10
- GDAL (system dependency)

### Local

```bash
git clone https://github.com/afterrealism/GIS-ML-Workflow.git
cd GIS-ML-Workflow
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Docker

```bash
docker build -t mlwkf .
docker run mlwkf
```

### NCI Gadi

```bash
ssh <USERNAME>@gadi.nci.org.au

module purge
module load pbs python3-as-python gdal/3.0.2

export MLHOME=/g/data/ge3/$USER
mkdir -p $MLHOME/github && cd $MLHOME/github
git clone https://github.com/afterrealism/GIS-ML-Workflow.git
cd GIS-ML-Workflow

python3 -m venv $MLHOME/venvs/mlwkf
source $MLHOME/venvs/mlwkf/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Start Ray (required for distributed stages)
ray start --head

# Run the full pipeline
python -m mlwkf -c configurations_examples/reference_configuration.ini

# Run with custom log level
python -m mlwkf -c config.ini -l DEBUG
```

### Configuration

The pipeline is driven by a single `.ini` config file. See `configurations_examples/` for examples.

Key sections:

| Section | Purpose |
|---------|---------|
| `[Workflow]` | Toggle each stage on/off |
| `[Model]` | Model name and parameters |
| `[Target]` | Target vector file, property name, OOS split |
| `[Covariates]` | List of GeoTIFF covariate paths |
| `[FeatureExtraction]` | Algorithm and settings |
| `[HyperParameterOptimization]` | HPO algorithm, search space, scoring |
| `[OutputFolder]` | Output directory path |
| `[Control]` | CPUs/GPUs per job |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAY_REDIS_PASSWORD` | `5241590000000000` | Ray Redis authentication |

## Testing

```bash
# Run all tests
pytest -rx -s tests

# Run specific test file
pytest -rx -s tests/test_evaluation_metrics.py

# Run with coverage
pytest --cov=mlwkf --cov-report=term-missing tests
```

## License

See [LICENSE](LICENSE).
