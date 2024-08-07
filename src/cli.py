import argparse
import os
import textwrap
from dataclasses import dataclass

import toml

from remote_training.training_script import TrainingScript
from src.training_job import TrainingJob


class CLI:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Train a Yolov8 model with Vertex AI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(self._show_config_example()),
        )
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="The path to the config file.",
        )
        self.parser.add_argument(
            '--local',
            action=argparse.BooleanOptionalAction,
            help="Run the trainer job in your local machine"
        )

    def run(self):
        args = self.parser.parse_args()
        config = self._parse_toml(args.config)
        training_config = self._training_config(config)
        if args.local:
            print("\n-------------------------------\n")
            print("Running in a local machine")
            print("\n-------------------------------\n")
            self._run_local(config, training_config)
        else:
            print("\n-------------------------------\n")
            print("Running in a remote machine")
            print("\n-------------------------------\n")
            self._run_remote(config, training_config)

    def _show_config_example(self) -> str:
        toml_file_example = "config_example.toml"
        config = self._parse_toml(toml_file_example)
        return f"Config example:\n\n{toml.dumps(config)}"

    def _parse_toml(self, file_path) -> dict:
        with open(file_path, "r") as f:
            config = toml.load(f)
        return config

    def _training_config(self, config):
        return TrainingConfig(
            label_studio_token=config["label_studio"]["token"],
            label_studio_url=config["label_studio"]["url"],
            label_studio_project_id=config["label_studio"]["project_id"],
            image_size=config["training"]["image_size"],
            epochs=config["training"]["epochs"],
            model=config["training"]["model"],
            obb=config["training"]["obb"],
            number_of_folds=config["training"]["number_of_folds"],
            use_kfold=config["training"]["use_kfold"],
            accelerator_count=config["vertex_ai_machine_config"]["accelerator_count"],
            mlflow_tracking_uri=config["mlflow"]["tracking_uri"],
            mlflow_experiment_name=config["mlflow"]["experiment_name"],
            mlflow_model_name=config["mlflow"]["model_name"],
            mlflow_run=config["mlflow"]["run"],
            mlflow_tracking_username=config["mlflow"]["user"],
            mlflow_tracking_password=config["mlflow"]["password"],
            source_images_bucket=config["google_cloud"]["source_images_bucket"],
            source_images_directory=config["google_cloud"]["source_images_directory"],
            trained_models_bucket=config["google_cloud"]["trained_models_bucket"],
        )

    def _run_remote(self, config, training_config):
        train_job = TrainingJob(
            gc_project=config["google_cloud"]["project"],
            gc_bucket=config["google_cloud"]["trained_models_bucket"],
            machine_type=config["vertex_ai_machine_config"]["machine_type"],
            accelerator_type=config["vertex_ai_machine_config"]["accelerator_type"],
            accelerator_count=config["vertex_ai_machine_config"]["accelerator_count"],
            training_config=training_config,
        )
        train_job.run()

    def _run_local(self, _config, training_config):
        env_vars = self._load_environment_variables(training_config)
        for key, value in env_vars.items():
            os.environ[key] = str(value)

        train_script = TrainingScript()
        train_script.run()

    def _load_environment_variables(self, training_config):
        # This code is repeated with #_load_environment_variables in TrainingJob
        return {
            "IMAGE_SIZE": str(training_config.image_size),
            "EPOCHS": str(training_config.epochs),
            "MODEL": str(training_config.model),
            "OBB": str(training_config.obb),
            "LABEL_STUDIO_TOKEN": str(training_config.label_studio_token),
            "LABEL_STUDIO_URL": str(training_config.label_studio_url),
            "LABEL_STUDIO_PROJECT_ID": str(training_config.label_studio_project_id),
            "NUMBER_OF_FOLDS": str(training_config.number_of_folds),
            "USE_KFOLD": str(training_config.use_kfold),
            "ACCELERATOR_COUNT": str(training_config.accelerator_count),
            "MLFLOW_TRACKING_URI": str(training_config.mlflow_tracking_uri),
            "MLFLOW_EXPERIMENT_NAME": str(training_config.mlflow_experiment_name),
            "MLFLOW_MODEL_NAME": str(training_config.mlflow_model_name),
            "MLFLOW_RUN": str(training_config.mlflow_run),
            "MLFLOW_TRACKING_USERNAME": str(training_config.mlflow_tracking_username),
            "MLFLOW_TRACKING_PASSWORD": str(training_config.mlflow_tracking_password),
            "SOURCE_IMAGES_BUCKET": str(training_config.source_images_bucket),
            "SOURCE_IMAGES_DIRECTORY": str(training_config.source_images_directory),
            "TRAINED_MODELS_BUCKET": str(training_config.trained_models_bucket),
        }


@dataclass
class TrainingConfig:
    image_size: str
    epochs: int
    model: str
    obb: bool
    label_studio_token: str
    label_studio_url: str
    label_studio_project_id: str
    number_of_folds: int
    use_kfold: bool
    accelerator_count: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    mlflow_run: str
    mlflow_tracking_username: str
    mlflow_tracking_password: str
    source_images_bucket: str
    source_images_directory: str
    trained_models_bucket: str


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
