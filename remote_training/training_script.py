import datetime
import logging
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml

import ultralytics.utils
from ultralytics import YOLO
from label_studio_sdk import Client as LabelStudioClient


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

        self.source_images_directory = Path(os.environ["SOURCE_IMAGES_DIRECTORY"])
        self.trained_models_bucket_name = os.environ['TRAINED_MODELS_BUCKET']

    def run(self):
        # Number of contiguous images
        image_group_size = 100
        if not self.use_mlflow:
            self._turn_off_mlflow_logging_on_yolo()

        class_names, annotations = self._download_dataset_annotations()
        images = self._download_labeled_dataset_images()

        dataset_path = self.single_dataset_path
        dataset_yaml = self._create_single_dataset(annotations, class_names, images, dataset_path,
                                                   self.validation_percentage, image_group_size)
        model_name = "single_model"
        model = self._train_model(dataset_yaml, model_name)
        self._save_model_metrics(model_name, model)

    # PRIVATE

    def _define_save_path(self):
        formatted_datetime = datetime.datetime.now().isoformat().replace('.', '').replace(':', '')
        return Path(self.dataset_path / f"{formatted_datetime}_Single_Model_Training")

    # GPU

    def _check_if_gpu_is_available(self):
        gpu_available = torch.cuda.is_available()
        logging.info(f"Checking if GPU is available: {gpu_available}")
        if self.accelerator_count > 0 and not gpu_available:
            logging.error(f"GPU is not available, accelerator count: {self.accelerator_count}")
            exit(1)
        return gpu_available

    # Dataset

    def _download_dataset_annotations(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)

        labels_path = self.dataset_path / "labels"
        for label_file in labels_path.glob("*.txt"):
            new_label_filename = label_file.name.split("__", 1)[-1]
            label_file.rename(labels_path / new_label_filename)

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
        labeled_tasks = self.label_studio_project.get_labeled_tasks() # TODO: CAGADA ACA
        labeled_image_names = list(map(self.extract_image_name, labeled_tasks))

        all_dataset_image_paths = []
        for image_name in labeled_image_names:
            destination_image_path = self.dataset_path / image_name
            all_dataset_image_paths.append(destination_image_path)

        return sorted(all_dataset_image_paths)

    def _create_single_dataset(self, annotations, class_names, images, datasets_path, val_percentage, image_group_size):
        folder_name = 'single_dataset'
        model_info_dir = datasets_path / folder_name
        model_info_dir.mkdir(parents=True, exist_ok=True)

        # Crear estructura de directorios
        for subset in ['train', 'val', 'test']:
            (model_info_dir / subset / "images").mkdir(parents=True, exist_ok=True)
            (model_info_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

        # Crear archivo YAML
        dataset_yaml = model_info_dir / "dataset.yaml"
        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": f"{self.base_path}/{model_info_dir.as_posix()}",
                    "train": "train",
                    "val": "val",
                    "test": "test",
                    "names": class_names,
                },
                ds_y,
            )

        num_images = len(images)
        num_groups = math.ceil(num_images / image_group_size)

        for group_idx in range(num_groups):
            start_idx = group_idx * image_group_size
            end_idx = min((group_idx + 1) * image_group_size, num_images)
            group_images = images[start_idx:end_idx]
            group_anns = annotations[start_idx:end_idx]

            group_size = len(group_images)

            # Cálculo preciso garantizando al menos 1 imagen en val y test si group_size >= 7
            train_size = round(group_size * 0.8)
            val_size = round(group_size * 0.15)
            test_size = group_size - train_size - val_size

            # Ajuste para asegurar que todos los splits tengan imágenes
            if val_size == 0 and group_size > 1:
                train_size -= 1
                val_size = 1
            if test_size == 0 and group_size > 2:
                train_size -= 1
                test_size = 1

            # Dividir los grupos
            train_images = group_images[:train_size]
            train_anns = group_anns[:train_size]

            val_images = group_images[train_size:train_size+val_size]
            val_anns = group_anns[train_size:train_size+val_size]

            test_images = group_images[train_size+val_size:]
            test_anns = group_anns[train_size+val_size:]

            # Función auxiliar para copiar archivos
            def copy_files(images, annotations, subset):
                for image, label in zip(images, annotations):
                    image_name = image.name.split("__", 1)[-1]
                    label_name = label.name.split("__", 1)[-1]
                    shutil.copy(image, model_info_dir / subset / "images" / image_name)
                    shutil.copy(label, model_info_dir / subset / "labels" / label_name)

            # Copiar a los directorios correspondientes
            copy_files(train_images, train_anns, 'train')
            copy_files(val_images, val_anns, 'val')
            copy_files(test_images, test_anns, 'test')

        # Verificación final
        print(f"Distribución final:")
        print(f"- Train: {len(list((model_info_dir / 'train' / 'images').glob('*')))} imágenes")
        print(f"- Val: {len(list((model_info_dir / 'val' / 'images').glob('*')))} imágenes")
        print(f"- Test: {len(list((model_info_dir / 'test' / 'images').glob('*')))} imágenes")
    
        return dataset_yaml

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
            "hsv_h": 0.05,
            "hsv_s": 0.5,
            "hsv_v": 0.7,
            "degrees": 12.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 3.0,
            "perspective": 0.0001,
            "flipud": 0.0,
            "fliplr": 0.5,
            "bgr": 0.0,
            "mosaic": 0.5,
            "mixup": 0.1,
            "cutmix": 0.5,
            "erasing": 0.4
        }
        return augmentations

    def _get_device(self):
        return '0' if self._check_if_gpu_is_available() else 'cpu'

    # Exporting results

    def _save_model_metrics(self, fold_name, model):
        metrics = model.val(split="test", single_cls=True, plots=True, visualize=True)
        results = pd.DataFrame(
            {
                "p": metrics.box.p,
                "r": metrics.box.r,
                "map50": metrics.box.ap50,
                "map50-95": metrics.box.ap,
            }
        )
        results.to_csv(f"{self.training_results_path}/{fold_name}/metrics.csv")

    def _turn_off_mlflow_logging_on_yolo(self):
        # This is a hacky way to avoid ultralytics using mlflow logging when we don't want it
        ultralytics.utils.TESTS_RUNNING = True


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
