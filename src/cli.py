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
            label_studio_token=config["label_studio"].get("token"),
            label_studio_url=config["label_studio"].get("url"),
            label_studio_project_id=config["label_studio"].get("project_id"),
            image_size=config["training"].get("image_size"),
            epochs=config["training"].get("epochs"),
            model=config["training"].get("model"),
            obb=config["training"].get("obb"),
            number_of_folds=config["training"].get("number_of_folds"),
            use_kfold=config["training"].get("use_kfold"),
            accelerator_count=config["vertex_ai_machine_config"].get("accelerator_count"),
            use_mlflow=config["mlflow"].get("use_mlflow"),
            mlflow_tracking_uri=config["mlflow"].get("tracking_uri"),
            mlflow_experiment_name=config["mlflow"].get("experiment_name"),
            mlflow_model_name=config["mlflow"].get("model_name"),
            mlflow_run=config["mlflow"].get("run"),
            mlflow_tracking_username=config["mlflow"].get("user"),
            mlflow_tracking_password=config["mlflow"].get("password"),
            source_images_bucket=config["google_cloud"].get("source_images_bucket"),
            source_images_directory=config["google_cloud"].get("source_images_directory"),
            trained_models_bucket=config["google_cloud"].get("trained_models_bucket"),
            validation_percentage=config["training"].get("validation_percentage"),
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

    def _run_local(self, config, training_config):
        env_vars = self._load_environment_variables(training_config)
        for key, value in env_vars.items():
            os.environ[key] = str(value)

        self._configure_service_account_if_neccesary(config)
        train_script = TrainingScript()
        train_script.run()

    def _configure_service_account_if_neccesary(self, config):
        if config['google_cloud'].get('use_service_account', False):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['google_cloud']['service_account']

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
            "ACCELERATOR_COUNT": str(training_config.accelerator_count),
            "USE_KFOLD": str(training_config.use_kfold),
            "USE_MLFLOW": str(training_config.use_mlflow),
            "MLFLOW_TRACKING_URI": str(training_config.mlflow_tracking_uri),
            "MLFLOW_EXPERIMENT_NAME": str(training_config.mlflow_experiment_name),
            "MLFLOW_MODEL_NAME": str(training_config.mlflow_model_name),
            "MLFLOW_RUN": str(training_config.mlflow_run),
            "MLFLOW_TRACKING_USERNAME": str(training_config.mlflow_tracking_username),
            "MLFLOW_TRACKING_PASSWORD": str(training_config.mlflow_tracking_password),
            "SOURCE_IMAGES_BUCKET": str(training_config.source_images_bucket),
            "SOURCE_IMAGES_DIRECTORY": str(training_config.source_images_directory),
            "TRAINED_MODELS_BUCKET": str(training_config.trained_models_bucket),
            "VALIDATION_PERCENTAGE": int(training_config.validation_percentage)
        }


@dataclass
class TrainingConfig:
    number_of_folds: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    mlflow_run: str
    mlflow_tracking_username: str
    mlflow_tracking_password: str
    image_size: str
    epochs: int
    model: str
    label_studio_token: str
    label_studio_url: str
    label_studio_project_id: str
    accelerator_count: int
    source_images_bucket: str
    source_images_directory: str
    trained_models_bucket: str
    validation_percentage: int
    obb: bool = False
    use_kfold: bool = False
    use_mlflow: bool = False

    def __post_init__(self):
        required_fields = ["image_size", "epochs", "model", "label_studio_token", "label_studio_url",
                           "label_studio_project_id", "accelerator_count", "source_images_bucket",
                           "source_images_directory", "trained_models_bucket", "validation_percentage"]

        for field in required_fields:
            attr = self.__getattribute__(field)
            if attr is None:
                raise ValueError(f"{field} is required and must be set by the user.")


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
