# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# below. Please refer to the the TensorFlow dockerfiles documentation for
# more information. Build args are documented as their default value.
#
# Ubuntu-based, Nvidia-GPU-enabled environment for using TensorFlow, with Jupyter included.
#
# NVIDIA with CUDA and CuDNN, no dev stuff
# --build-arg UBUNTU_VERSION=16.04
#    ( no description )
#
# Python is required for TensorFlow and other libraries.
# --build-arg USE_PYTHON_3_NOT_2=True
#    Install python 3 over Python 2
#
# Install the TensorFlow Python package.
# --build-arg TF_PACKAGE=tensorflow-gpu (tensorflow|tensorflow-gpu|tf-nightly|tf-nightly-gpu)
#    The specific TensorFlow Python package to install
#
# Configure TensorFlow's shell prompt and login tools.
#
# Launch Jupyter on execution instead of a bash prompt.

FROM nvidia/cuda:9.0-base-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
        apt-get update && \
        apt-get install libnvinfer4=4.1.2-1+cuda9.0

RUN apt-get install vim -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get update && apt-get install -y curl wget
RUN apt-get install python3-pip -y

RUN pip3 install --upgrade pip setuptools
RUN pip3 install tensorflow-gpu tensorflowjs
RUN pip3 install jupyter virtualenv
RUN pip3 uninstall --yes prompt_toolkit
RUN pip3 install prompt_toolkit==2.0.4

# RUN apt-get remove python -y
# RUN apt-get remove python2.7 -y
# RUN apt-get remove python2.7-minimal -y
# RUN apt-get remove python3 -y
# RUN apt-get remove python3.5 -y
# RUN apt-get remove python3.5-minimal -y

# https://stackoverflow.com/questions/43759610/how-to-add-python-3-6-kernel-alongside-3-5-on-jupyter

RUN mkdir /notebooks && chmod a+rwx /notebooks
RUN mkdir /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888
EXPOSE 6006

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# RUN rm -f /usr/bin/python3
# RUN ln -s /usr/bin/python3.6 /usr/bin/python3
# RUN ln -s /usr/bin/python3.6 /usr/bin/python

# CMD ["bash", "-c", "tensorboard --logdir=/logs"]
# CMD ["bash", "-c", "jupyter notebook -i 0.0.0.0"]
CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"]
