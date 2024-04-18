import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            ##################
            _ = mlflow.run(
                os.path.join(root_path, "basic_cleaning"),
                "main",
                parameters = {
                    "input_artifact": "raw_data.csv:latest",
                    "output_artifact": "clean_data.csv"
                },
            )
            ##################
            pass

        if "data_check" in active_steps:
            ##################
            _ = mlflow.run(
                os.path.join(root_path, "data_check"), 
                "main", 
                parameters = {
                    "reference_artifact": config["data"]["reference_dataset"], 
                    "sample_artifact": "clean_data.csv:latest", 
                    "ks_alpha": config["data"]["ks_alpha"]
                },
            )
            ##################
            pass

        if "data_split" in active_steps:
            ##################
            _ = mlflow.run(
                os.path.join(root_path, "segregate"),
                "main",
                parameters={
                    "input_artifact": "preprocessed_data.csv:latest",
                    "artifact_root": "data",
                    "artifact_type": "segregated_data",
                    "test_size": config["data"]["test_size"],
                    "stratify": config["data"]["stratify"]
                },
            )
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest step

            ##################
            _ = mlflow.run(
                os.path.join(root_path, "random_forest"),
                "main",
                parameters={
                    "train_data": "data_train.csv:latest",
                    "rf_config": rf_config,
                    "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                    "random_seed": config["main"]["random_seed"],
                    "val_size": config["data"]["test_size"],
                    "stratify": config["data"]["stratify"]
                },
            )
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            _ = mlflow.run(
                os.path.join(root_path, "evaluate"),
                "main",
                parameters={
                    "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                    "test_data": "data_test.csv:latest"
                },
            )
            ##################

            pass


if __name__ == "__main__":
    go()
