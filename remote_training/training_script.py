import datetime
import gc
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold
from tqdm import tqdm

# This is a workaround to avoid Ultralytics trying to do multi-gpu training. Despite trying to set it as
# an env var, it still tries to do multi-gpu training. This is a workaround to avoid that.
# It's necessary to do this before importing ultralytics
os.environ["RANK"] = "-1"
import ultralytics.utils
from ultralytics import YOLO
from label_studio_sdk.converter import Converter
from label_studio_sdk import Client as LabelStudioClient
from google.cloud import storage


class TrainingScript:
    def __init__(self) -> None:
        self.image_size = [int(size) for size in os.environ["IMAGE_SIZE"].replace(" ", "").split(",")]
        self.epochs = int(os.environ["EPOCHS"])
        self.model = os.environ["MODEL"]
        self.obb = os.environ["OBB"] == "True"
        self.base_path = os.getcwd()
        self.dataset_path = Path("dataset")
        self.number_folds = int(os.environ["NUMBER_OF_FOLDS"])
        self.use_kfold = (os.environ["USE_KFOLD"] == "True")
        self.save_path = self._define_save_path()
        self.training_results_path = self.save_path / "training_results"
        self.fold_datasets_path = self.save_path / "folds_datasets"
        self.single_dataset_path = self.save_path / "single_dataset"
        self.validation_percentage = os.environ["VALIDATION_PERCENTAGE"]

        self.use_mlflow = (os.environ["USE_MLFLOW"] == "True")
        self.mlflow_model_name = os.environ["MLFLOW_MODEL_NAME"]
        self.mlflow_experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

        self.accelerator_count = int(os.environ["ACCELERATOR_COUNT"])
        self.rank = os.environ["RANK"]

        self.label_studio_url = os.environ["LABEL_STUDIO_URL"]
        self.label_studio_token = os.environ["LABEL_STUDIO_TOKEN"]
        self.label_studio_project_id = int(os.environ["LABEL_STUDIO_PROJECT_ID"])
        label_studio = LabelStudioClient(url=self.label_studio_url, api_key=self.label_studio_token)
        self.label_studio_project = label_studio.get_project(self.label_studio_project_id)

        google_cloud_client = storage.Client()
        self.source_images_bucket = google_cloud_client.get_bucket(os.environ["SOURCE_IMAGES_BUCKET"])
        self.source_images_directory = Path(os.environ["SOURCE_IMAGES_DIRECTORY"])
        self.trained_models_bucket_name = os.environ['TRAINED_MODELS_BUCKET']

    def run(self):
        if not self.use_mlflow:
            self._turn_off_mlflow_logging_on_yolo()

        class_names, annotations = self._download_dataset_annotations()
        images = self._download_labeled_dataset_images()

        if self.use_kfold:
            dataset_path = self.fold_datasets_path
            dataset_yaml_list = self._create_k_folds(annotations, class_names, images, dataset_path)
            for fold_number in range(len(dataset_yaml_list)):
                dataset_yaml = dataset_yaml_list[fold_number]
                model_name = self._fold_name(fold_number)
                model = self._train_model(dataset_yaml, model_name)
                self._add_information_to_model_in_mlflow_if_neccesary()
                self._save_model_metrics(model_name, model)
                self._clean_gpu_cache()  # This is necessary to avoid running out of memory
        else:
            dataset_path = self.single_dataset_path
            dataset_yaml = self._create_single_dataset(annotations, class_names, images, dataset_path,
                                                       self.validation_percentage)
            model_name = "single_model"
            model = self._train_model(dataset_yaml, model_name)
            self._add_information_to_model_in_mlflow_if_neccesary()
            self._save_model_metrics(model_name, model)

        self._export_results()

    # PRIVATE

    def _define_save_path(self):
        formatted_datetime = datetime.datetime.now().isoformat().replace('.', '').replace(':', '')
        if self.use_kfold:
            return Path(self.dataset_path / f"{formatted_datetime}_{self.number_folds}-Fold_Cross-val")
        else:
            return Path(self.dataset_path / f"{formatted_datetime}_Single_Model_Training")

    # GPU

    def _check_if_gpu_is_available(self):
        gpu_available = torch.cuda.is_available()
        logging.info(f"Checking if GPU is available: {gpu_available}")
        if self.accelerator_count > 0 and not gpu_available:
            logging.error(f"GPU is not available, accelerator count: {self.accelerator_count}")
            exit(1)
        return gpu_available

    def _prevent_multi_gpu_training(self):
        logging.info(f"Checking RANK: {self.rank}")
        if self.rank != "-1":
            logging.error("Trying multi gpu training. Exiting.")
            exit(1)

    def _clean_gpu_cache(self):
        if self._check_if_gpu_is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Dataset

    def _download_dataset_annotations(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)

        if self.obb:
            json_annotations_path = self.dataset_path / 'annotations.json'
            self._export_annotations_from_label_studio("JSON", json_annotations_path)
            self._convert_annotations_into_yolo_obb(json_annotations_path, self.dataset_path)
        else:
            yolo_annotations_path = self.dataset_path / "annotations.zip"
            self._export_annotations_from_label_studio("YOLO", yolo_annotations_path)
            shutil.unpack_archive(yolo_annotations_path, extract_dir=self.dataset_path)

        with open(f"{self.dataset_path}/classes.txt", "r") as f:
            class_names = f.read().splitlines()
        yaml_data = {
            "names": class_names,
            "nc": len(class_names),
            "train": f"{self.base_path}/{self.dataset_path}/train",
            "val": f"{self.base_path}/{self.dataset_path}/val",
        }
        yaml_file_path = f"{self.dataset_path}/data.yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        labels_path = self.dataset_path / "labels"
        annotations = sorted(labels_path.rglob("*.txt"))
        return class_names, annotations

    def _export_annotations_from_label_studio(self, export_type, output_path):
        os.system(
            f"curl -X GET {self.label_studio_url}api/projects/{self.label_studio_project_id}/export\?exportType\={export_type} \
            -H 'Authorization: Token {self.label_studio_token}' --output '{str(output_path)}'"
        )

    def _convert_annotations_into_yolo_obb(self, json_annotations_path, output_dir_path):
        label_config = self.label_studio_project.parsed_label_config

        converter = Converter(config=label_config, project_dir='.')
        converter.convert_to_yolo(
            input_data=str(json_annotations_path),
            is_dir=False,
            output_dir=str(output_dir_path),
            is_obb=True)

    def extract_image_name(self, task):
        # TODO REVISAR ESTO
        import base64
        from urllib.parse import urlparse, parse_qs

        image_url = task["data"]["image"]
        parsed_url = urlparse(image_url)
        query = parse_qs(parsed_url.query)

        if "fileuri" in query:
            # Decodificar de base64 a string
            fileuri_b64 = query["fileuri"][0]
            fileuri = base64.b64decode(fileuri_b64).decode("utf-8")
            # Obtener solo el nombre del archivo
            return Path(fileuri).name
        else:
            # Fallback si no hay fileuri
            return Path(parsed_url.path).name

    def _download_labeled_dataset_images(self):
        labeled_tasks = self.label_studio_project.get_labeled_tasks()
        labeled_image_names = list(map(self.extract_image_name, labeled_tasks))

        all_dataset_image_paths = []
        for image_name in tqdm(labeled_image_names):
            source_image_path = self.source_images_directory / image_name
            destination_image_path = self.dataset_path / image_name
            all_dataset_image_paths.append(destination_image_path)

            google_cloud_image = self.source_images_bucket.blob(str(source_image_path))
            google_cloud_image.download_to_filename(destination_image_path)

        return sorted(all_dataset_image_paths)

    def _create_single_dataset(self, annotations, class_names, images, datasets_path, val_percentage):
        folder_name = 'single_dataset'
        model_info_dir = datasets_path / folder_name
        model_info_dir.mkdir(parents=True, exist_ok=True)
        (model_info_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = model_info_dir / f"dataset.yaml"

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": f"{self.base_path}/{model_info_dir.as_posix()}",
                    "train": "train",
                    "val": "val",
                    "names": class_names,
                },
                ds_y,
            )

        images_to_move = max(int(images * val_percentage / 100), 1) # At least one image for validation
        random_image_indexes = random.sample(range(0, len(images) - 1), images_to_move)
        for idx, (image, label) in enumerate(zip(images, annotations)):
            val_or_train_folder = 'val' if idx in random_image_indexes else 'train'
            image_name = image.name.split("__", 1)[-1]
            label_name = label.name.split("__", 1)[-1]
            shutil.copy(image, datasets_path / folder_name / val_or_train_folder / 'images' / image_name)
            shutil.copy(label, datasets_path / folder_name / val_or_train_folder / 'labels' / label_name)

        return dataset_yaml

    def _create_k_folds(self, annotations, class_names, images, datasets_path):
        file_names = [annotation.stem for annotation in annotations]
        class_indices = list(range(len(class_names)))

        labels_per_image = self._build_df_with_labels_per_image(annotations, class_indices, file_names)

        # It's necessary to cast to list because a generator can only be used once, and we use it many times.
        k_fold_indices = list(KFold(n_splits=self.number_folds, shuffle=True, random_state=20).split(labels_per_image))

        fold_names = [self._fold_name(n) for n in range(self.number_folds)]
        folds_df = self._build_df_with_image_distribution_for_folds(k_fold_indices, labels_per_image, file_names,
                                                                    fold_names)
        fold_label_distribution = self._build_df_with_label_distribution_for_folds(fold_names, class_indices,
                                                                                   k_fold_indices, labels_per_image)
        folds_yamls = self._build_train_and_val_dataset_yaml(folds_df, class_names, datasets_path)
        self._build_datasets_folders(folds_df, datasets_path)
        self._copy_images_and_labels_to_datasets(annotations, datasets_path, folds_df, images)
        folds_df.to_csv(datasets_path / "kfold_datasplit.csv")
        fold_label_distribution.to_csv(datasets_path / "kfold_label_distribution.csv")
        return folds_yamls

    def _fold_name(self, k):
        return f"fold_{k + 1}"

    def _build_df_with_labels_per_image(self, annotations, class_indices, file_names):
        labels_per_image = pd.DataFrame([], columns=class_indices, index=file_names)
        for label in annotations:
            label_counter = Counter()
            with open(label, "r") as lf:
                lines = lf.readlines()
            for line in lines:
                label_counter[int(line.split(" ")[0])] += 1
            labels_per_image.loc[label.stem] = label_counter
        return labels_per_image.fillna(0.0)

    def _build_df_with_image_distribution_for_folds(self, k_fold_indices, labels_per_image, file_names, fold_names):
        folds_df = pd.DataFrame(index=file_names, columns=fold_names)

        for index, (train, val) in enumerate(k_fold_indices):
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[train].index] = "train"
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[val].index] = "val"
        return folds_df

    def _build_df_with_label_distribution_for_folds(self, fold_names, class_indices, k_fold_indices, labels_per_image):
        fold_label_distribution = pd.DataFrame(index=fold_names, columns=class_indices)
        for n, (train_indices, val_indices) in enumerate(k_fold_indices):
            train_totals = labels_per_image.iloc[train_indices].sum()
            val_totals = labels_per_image.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1e-7)
            fold_label_distribution.loc[self._fold_name(n)] = ratio
        return fold_label_distribution

    def _build_train_and_val_dataset_yaml(self, folds_df, class_names, datasets_path):
        folds_yamls = []
        for fold in folds_df.columns:
            fold_dir = datasets_path / fold
            fold_dir.mkdir(parents=True, exist_ok=True)
            dataset_yaml = fold_dir / f"dataset.yaml"
            with open(dataset_yaml, "w") as ds_y:
                yaml.safe_dump(
                    {
                        "path": f"{self.base_path}/{fold_dir.as_posix()}",
                        "train": "train",
                        "val": "val",
                        "names": class_names,
                    },
                    ds_y,
                )
            folds_yamls.append(dataset_yaml)
        return folds_yamls

    def _build_datasets_folders(self, folds_df, datasets_path):
        for fold in folds_df.columns:
            fold_dir = datasets_path / fold
            (fold_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    def _copy_images_and_labels_to_datasets(self, annotations, datasets_path, folds_df, images):
        for image, label in zip(images, annotations):
            for fold, train_or_val in folds_df.loc[image.stem].items():
                shutil.copy(image, datasets_path / fold / train_or_val / "images" / image.name)
                shutil.copy(label, datasets_path / fold / train_or_val / "labels" / label.name)

    # Training

    def _train_model(self, dataset_yaml, model_name):
        augmentations = self._augmentations()
        model = YOLO(self.model)
        model.train(
            data=dataset_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            rect=(self.image_size[0] != self.image_size[1]),
            device=self._get_device(),
            project=str(self.training_results_path),
            name=model_name,
            **augmentations,
        )
        return model

    def _augmentations(self):
        augmentations = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 4.0,
            "translate": 0.1,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
        return augmentations

    def _get_device(self):
        return '0' if self._check_if_gpu_is_available() else 'cpu'

    # Exporting results

    def _save_model_metrics(self, fold_name, model):
        metrics = model.metrics.box
        results = pd.DataFrame(
            {
                "p": metrics.p,
                "r": metrics.r,
                "map50": metrics.ap50,
                "map50-95": metrics.ap,
            }
        )
        results.to_csv(f"{self.training_results_path}/{fold_name}/metrics.csv")

    def _export_results(self):
        os.system(f'gsutil -m cp -r "{self.save_path}" "gs://{self.trained_models_bucket_name}"')

    # Mlflow

    def _add_information_to_model_in_mlflow_if_neccesary(self):
        if self.use_mlflow:
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            last_run_id = self._get_last_run_id_for(experiment)
            with mlflow.start_run(run_id=last_run_id):
                # It is restarting the run that YOLO started. We want to add some information to it
                mlflow.log_params(self.__dict__)

            model_uri = self._get_model_uri_for(last_run_id)
            mlflow.register_model(model_uri, self.mlflow_model_name)

    def _get_model_uri_for(self, last_run_id):
        model_uri = f"runs:/{last_run_id}/artifacts/weights/best.pt"
        return model_uri

    def _get_last_run_id_for(self, experiment):
        runs_ordered_by_end_time = mlflow.search_runs([experiment.experiment_id], order_by=["end_time DESC"])
        last_run_id = runs_ordered_by_end_time.loc[0, 'run_id']
        return last_run_id

    def _turn_off_mlflow_logging_on_yolo(self):
        # This is a hacky way to avoid ultralytics using mlflow logging when we don't want it
        ultralytics.utils.TESTS_RUNNING = True


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
