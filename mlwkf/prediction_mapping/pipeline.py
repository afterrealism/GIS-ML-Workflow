import os
import time
import logging
import ray
import configparser
from pathlib import Path, PosixPath

from mlwkf.registry import lookup_model, safe_parse_list
from mlwkf.prediction_mapping.utilities import get_extent_coordinates, create_predicted_geotiff


def run_prediction_pipeline(config_file_path):

    os.environ["MODIN_ENGINE"] = "ray"
    time.sleep(2.0)

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    cpus_per_job = config.getint('Control', 'cpus_per_job')
    gpu_per_job = config.getfloat('Control', 'gpu_per_job')

    output_folder = Path(list(config['OutputFolder'].keys())[0])
    output_folder.mkdir(parents=True, exist_ok=True)

    line_geotiff_folder = output_folder / "line_geotiff"
    line_geotiff_folder.mkdir(parents=True, exist_ok=True)

    merged_geotiff_folder = output_folder / "merged_geotiff"
    merged_geotiff_folder.mkdir(parents=True, exist_ok=True)

    selected_features = safe_parse_list(config.get('Intermediate', 'selected_features'))
    path_to_trained_model = Path(config.get('PredictionMapping', 'path_to_trained_model'))
    area_of_interest = Path(config.get('Target', 'area_of_interest'))
    model_function = lookup_model(config.get('Model', 'model_function'))

    if not path_to_trained_model.exists():
        logging.warning("path_to_trained_model: %s", path_to_trained_model)
        raise Exception("Please provide valid path to trained model.")

    if not area_of_interest.exists():
        logging.warning("area_of_interest: %s", area_of_interest)
        raise Exception("Please provide valid path to area of interest.")

    # TODO update logic for checking feature names
    covariates = []
    for selected_feature in selected_features:
        for covariate in safe_parse_list(config.get('Intermediate', 'covariates')):
            if selected_feature in str(covariate):
                covariates.append(covariate)
                break

    path_to_predicted_geotiff = create_predicted_geotiff(area_of_interest, covariates, path_to_trained_model, line_geotiff_folder, merged_geotiff_folder, output_folder, model_function, cpus_per_job, gpu_per_job)

    config['Workflow']['PredictionMapping'] = "False"
    config['Results']['path_to_predicted_geotiff'] = str(path_to_predicted_geotiff)

    logging.warning("Updating config file")
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
