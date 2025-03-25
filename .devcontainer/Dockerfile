FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt update -y && apt install -y git python3 python3-pip libgl1 libglib2.0-0
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN apt-get install apt-transport-https ca-certificates gnupg curl
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

COPY requirements.txt /workspaces/cv-vertex-ai-trainer/requirements.txt
RUN pip3 install -r /workspaces/cv-vertex-ai-trainer/requirements.txt

