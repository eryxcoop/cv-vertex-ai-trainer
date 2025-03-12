import glob
import json
import os
from pathlib import Path

import supervision as sv
from PIL import Image
from label_studio_sdk import Client
from label_studio_sdk.converter.imports.yolo import convert_yolo_to_ls
from tqdm import tqdm
from ultralytics import YOLO

LABEL_STUDIO_URL = 'https://label-studio.k8s.eryx.co'
API_KEY = 'api_key'
PROJECT_ID = 58  # Can be found in the URL of the project in Label Studio. You have to create a Label Studio project first before running script

IMAGES_BUCKET = 'source_images_bucket'  # Images bucket to push. You have to create a bucket before running script

MODEL_PATH = 'yolov8n.pt'

VIDEO_PATH = 'scripts/quilmes02.mkv'
FRAME_INTERVAL = 500  # Frames to skip between model running


class PreannotateVideo:
    def __init__(self, label_studio_url, label_studio_api_key, project_id, source_images_bucket, model_path, video_path,
                 frame_interval):
        self.label_studio = Client(url=label_studio_url, api_key=label_studio_api_key)
        self.project_id = project_id
        self.source_images_bucket = source_images_bucket
        self.model_path = model_path
        self.video_path = video_path
        self.frame_interval = frame_interval

    def run(self):
        images_directory = self._create_tmp_folder_with_images_from_video()
        model = YOLO(self.model_path)
        predictions_path = self._create_folder_with_images_and_yolo_annotations("tmp_path", model)

        # classes_as_list = self._get_classes_used_on_label_studio_annotations()
        # self._write_classes_file(classes_as_list, predictions_path)

        # annotations_filename = self._convert_yolo_annotations_to_label_studio(predictions_path)

        # self._push_annotations(filtered_tasks, annotations_filename)

    def _get_classes_used_on_label_studio_annotations(self):
        return self.label_studio.get_project(self.project_id).parsed_label_config['label']['labels']

    # PRIVATE
    def _get_images_from_bucket(self):
        local_images_directory = self._create_local_images_and_annotations_directory()
        self._copy_all_images(self.source_images_bucket, local_images_directory)
        return local_images_directory

    def _create_local_images_and_annotations_directory(self):
        local_images_directory = Path("all_images")
        local_images_directory.mkdir(exist_ok=True)
        return local_images_directory

    def _copy_all_images(self, source_directory, destination_directory):
        os.system(f'gsutil -m cp -r "{source_directory}/*" {destination_directory}')

    def _get_unlabeled_tasks_from_project(self):
        source_project = self.label_studio.get_project(self.project_id)
        tasks_as_json = source_project.get_unlabeled_tasks()

        return tasks_as_json

    def _create_folder_with_images_and_yolo_annotations(self, images_directory, model):
        yolo_predictions_path = Path("yolo_predictions/prediction")
        images_yolo_predictions_path = yolo_predictions_path / "images"
        images_yolo_predictions_path.mkdir(parents=True, exist_ok=True)

        image_files = os.listdir(images_directory)

        # Iterate through images
        for image_file in image_files:
            image_path = f"{images_directory}/{image_file}"
            with Image.open(image_path) as image:

                predictions = model.predict(image)
                if self._has_at_least_one_prediction(predictions):
                    # TODO: estamos prediciendo dos veces por imagen, es para sacarlo andando pero mejorar.
                    model.predict(image, save_txt=True, project='yolo_predictions', name='prediction', exist_ok=True)
                    os.system(f'cp {str(image_path)} yolo_predictions/prediction/images/{str(image_path).split("/")[-1]}')
        return yolo_predictions_path

    def _has_at_least_one_prediction(self, predictions):
        return len(predictions[0].boxes.conf) > 0

    def _write_classes_file(self, classes_list, predictions_path):
        with open(f'{predictions_path}/classes.txt', 'w') as file:
            for a_class in classes_list:
                file.write(f'{a_class}\n')

    def _push_annotations(self, tasks_to_annotate, annotations_filename):
        source_project = self.label_studio.get_project(self.project_id)
        with open(annotations_filename, 'r') as annotations_file:
            annotations = json.load(annotations_file)

        dict_with_annotations = {}
        for annotation in annotations:
            name = annotation['data']['image'].split("/")[-1].split('.png')[0]
            if 'annotations' in annotation:
                dict_with_annotations[name] = annotation['annotations'][0]['result']

        for task in tasks_to_annotate:
            id_task = task['id']
            file_to_look = task["storage_filename"].split('/')[-1].split('.png')[0]
            if file_to_look in dict_with_annotations:
                source_project.create_annotation(id_task, result=dict_with_annotations[file_to_look])

    def _convert_yolo_annotations_to_label_studio(self, predictions_path):
        annotations_filename = 'annotations.json'
        convert_yolo_to_ls(str(predictions_path), annotations_filename)
        return annotations_filename

    def _create_tmp_folder_with_images_from_video(self):
        video_name = Path(self.video_path).stem
        image_name_pattern = video_name + "-{:05d}.png"
        tmp_path = "tmp_path"
        print("Obtaining images from video")
        with sv.ImageSink(target_dir_path=tmp_path, image_name_pattern=image_name_pattern) as sink:
            for image in tqdm(sv.get_video_frames_generator(source_path=self.video_path, stride=self.frame_interval)):
                sink.save_image(image=image)

        print("All images obtained")
        return tmp_path


if __name__ == "__main__":
    preannotate_video = PreannotateVideo(
        label_studio_url=LABEL_STUDIO_URL,
        label_studio_api_key=API_KEY,
        project_id=PROJECT_ID,
        source_images_bucket=IMAGES_BUCKET,
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        frame_interval=FRAME_INTERVAL
    )
    preannotate_video.run()
