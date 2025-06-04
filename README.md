# CV Vertex AI Trainer
CLI used to remotely train a Computer Vision model using the Google Vertex AI platform.

We are assuming:
- You have a Google Cloud Platform account
- You have a Google Cloud Platform project
- You have a Google Cloud Storage bucket
- You have a Label Studio project with images and annotations
- In order to retrieve your Label Studio token, refer to: https://api.labelstud.io/api-reference/introduction/getting-started#authentication
- Your images are stored in a Google Cloud Storage bucket

# Setup requirements

**Python version: 3.10**

1) Run `pip install -r requirements.txt`.

2) Install `gcloud CLI`, refer to: https://cloud.google.com/sdk/docs/install?hl=es-419

3) Run `gcloud auth login`.

    If you are getting a warning about "project quota" or about "default credentials not found" run `gcloud auth application-default login`.

4) Run `pip install .` to install the package (or `pip install -e .` to install in editable mode)

# Usage

Create a `.toml` file following the example on `config_example.toml`

Run `cv-vertex-ai-trainer -c config.toml` to start training on the cloud. 

Add `--local` to run the training locally.

## Training with OBB

When training with OBB (Oriented Bounding Box) you need to uncomment the line in `train_requirements.txt` that installs Ultralytics from a forked repository. This is TEMPORARY until the original repository fully supports OBB. 

Then you need to set the `obb` parameter to `true` in the configuration file, and pick a suitable YOLO model (for example `yolov8n-obb.pt`).