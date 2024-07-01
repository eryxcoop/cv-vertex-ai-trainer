import os
from pathlib import Path

from PIL import Image
from label_studio_sdk import Client

LABEL_STUDIO_URL = 'https://label-studio.example.co/'
API_KEY = 'api-key'
SOURCE_PROJECT_ID = 1  # Can be found in the URL of the project in Label Studio.
DESTINATION_PROJECT_ID = 2  # Can be found in the URL of the project in Label Studio.

SOURCE_IMAGES_BUCKET = 'gs://example'
DESTINATION_IMAGES_BUCKET = 'gs://example-resized'

ORIGINAL_IMAGE_WIDTH = 3024
ORIGINAL_IMAGE_HEIGHT = 4032
NEW_IMAGE_WIDTH = 1080
NEW_IMAGE_HEIGHT = 1440


class ResizeAnnotatedImages:
    def __init__(self, label_studio_url, label_studio_api_key, source_project_id, destination_project_id,
                 source_images_bucket, destination_images_bucket, new_image_width, new_image_height,
                 original_image_width, original_image_height):
        self.label_studio = Client(url=label_studio_url, api_key=label_studio_api_key)
        self.source_project_id = source_project_id
        self.destination_project_id = destination_project_id

        self.source_images_bucket = source_images_bucket
        self.destination_images_bucket = destination_images_bucket

        self.new_image_width = new_image_width
        self.new_image_height = new_image_height
        self.original_image_width = original_image_width
        self.original_image_height = original_image_height

    def run(self):
        self._resize_images_in_bucket()
        self._import_resized_images_into_label_studio()

    # PRIVATE

    def _resize_images_in_bucket(self):
        source_images_local_dir, resized_images_local_dir = self._create_local_image_directories()
        self._copy_all_images(self.source_images_bucket, source_images_local_dir)

        new_image_size = (self.new_image_width, self.new_image_height)
        self._resize_and_move_images(source_images_local_dir, resized_images_local_dir, new_image_size)

        self._copy_all_images(resized_images_local_dir, self.destination_images_bucket)

    def _import_resized_images_into_label_studio(self):
        source_project_tasks = self._get_all_tasks_from_source_project()
        updated_project_tasks = self._update_tasks_with_resized_images(source_project_tasks)
        self._import_tasks_to_destination_project(updated_project_tasks)

    def _create_local_image_directories(self):
        local_images_directory = Path("../images_to_resize")
        local_images_directory.mkdir(exist_ok=True)

        source_images_directory = local_images_directory.joinpath("original")
        source_images_directory.mkdir(exist_ok=True)

        resized_images_directory = local_images_directory.joinpath("resized")
        resized_images_directory.mkdir(exist_ok=True)

        return source_images_directory, resized_images_directory

    def _copy_all_images(self, source_directory, destination_directory):
        os.system(f'gsutil -m cp -r "{source_directory}/*" {destination_directory}')

    def _resize_and_move_images(self, source_images_dir, resized_images_dir, new_image_size):
        for image_path in source_images_dir.glob("*"):
            with Image.open(image_path) as image:
                resized_image = image.resize(new_image_size)
                resized_image_path = resized_images_dir.joinpath(image_path.name)
                resized_image.save(resized_image_path, quality=100)

    def _get_all_tasks_from_source_project(self):
        source_project = self.label_studio.get_project(self.source_project_id)
        tasks_as_json = source_project.get_tasks()

        return tasks_as_json

    def _update_tasks_with_resized_images(self, source_project_tasks):
        source_images_bucket_path = self.source_images_bucket.removeprefix("gs://")
        destination_images_bucket_path = self.destination_images_bucket.removeprefix("gs://")

        for task in source_project_tasks:
            resized_image_url = task["data"]["image"].replace(source_images_bucket_path, destination_images_bucket_path)
            task["data"]["image"] = resized_image_url
            for annotation in task["annotations"]:
                for result in annotation["result"]:
                    result["original_width"] = self.new_image_width
                    result["original_height"] = self.new_image_height

        return source_project_tasks

    def _import_tasks_to_destination_project(self, new_tasks):
        destination_project = self.label_studio.get_project(self.destination_project_id)
        destination_project.import_tasks(new_tasks)


if __name__ == "__main__":
    resize_annotated_images = ResizeAnnotatedImages(
        label_studio_url=LABEL_STUDIO_URL,
        label_studio_api_key=API_KEY,
        source_project_id=SOURCE_PROJECT_ID,
        destination_project_id=DESTINATION_PROJECT_ID,
        source_images_bucket=SOURCE_IMAGES_BUCKET,
        destination_images_bucket=DESTINATION_IMAGES_BUCKET,
        new_image_width=NEW_IMAGE_WIDTH,
        new_image_height=NEW_IMAGE_HEIGHT,
        original_image_width=ORIGINAL_IMAGE_WIDTH,
        original_image_height=ORIGINAL_IMAGE_HEIGHT
    )
    resize_annotated_images.run()
