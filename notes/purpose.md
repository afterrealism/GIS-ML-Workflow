# GIS-ML-Workflow (mlwkf) - Project Purpose

## What

A Python-based machine learning workflow automation tool for geoscience applications, originally developed at Geoscience Australia. It processes geospatial raster data (GeoTIFF) through a configurable, end-to-end ML pipeline.

## Why

Geoscientists need to run repeatable ML experiments on spatial datasets (e.g., mineral mapping, soil property prediction) without manually wiring together preprocessing, feature selection, hyperparameter tuning, and prediction steps each time. This tool automates the full pipeline via a single `.ini` configuration file.

## Pipeline Stages

1. **Data Preparation** - Ingests GeoTIFF covariates and target data, extracts training samples
2. **Data Exploration** - Generates summary statistics and visualizations of input data
3. **Feature Extraction** - Ranks and selects features using SHAP, recursive elimination, group CV, out-of-sample elimination, and randomness tests
4. **Hyperparameter Optimization** - Tunes model hyperparameters via Grid Search, Bayesian Search, HyperOpt, or HEBO
5. **Model Exploration** - Evaluates and compares model performance
6. **Prediction Mapping** - Generates spatial prediction maps from trained models
7. **Covariate Drift** - Detects distribution shifts between training and prediction data

## Supported Models

- XGBoost
- Random Forest
- SVR (Support Vector Regression)
- Bootstrapped and ensemble variants

## Target Environment

- NCI Gadi HPC (PBS job scheduler)
- Local Linux machines
- Uses Ray for distributed execution

## Author

Sheece Gardezi (sheecegardezi@gmail.com) - Geoscience Australia
