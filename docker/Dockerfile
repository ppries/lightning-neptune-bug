FROM nvcr.io/nvidia/tensorrt:22.04-py3

ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# Install core packages
RUN  apt-get update && \
     apt-get upgrade -y && \
     apt-get install -y --no-install-suggests \
      python3 \
      python3-dev \
      python3-pip

RUN python3 -m pip install -U pip

COPY docker/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --progress-bar off -U -r requirements.txt

COPY train.py /workspace/train.py

# Set CUDA environment variables
ENV CUDA_PATH=/usr/local/cuda
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.6"
ENV NVIDIA_VISIBLE_DEVICES=all
