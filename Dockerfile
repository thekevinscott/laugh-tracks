FROM nvidia/cuda:9.0-base-ubuntu16.04

######### Tensorflow Dependencies #########
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

######### System Packages #########
RUN apt-get install vim -y
RUN apt-get install -y git

######### Python #########
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get update && apt-get install -y curl wget
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip setuptools
RUN pip3 install tensorflow-gpu tensorflowjs
RUN pip3 install virtualenv
RUN pip3 uninstall --yes prompt_toolkit
RUN pip3 install prompt_toolkit==2.0.4

######### VGGish #########
RUN mkdir -p /lib
RUN git clone https://github.com/tensorflow/models /lib/models
RUN pip3 install numpy scipy
RUN pip3 install resampy tensorflow six
RUN curl -o /lib/models/research/audioset/vggish/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
RUN curl -o /lib/models/research/audioset/vggish/vggish_pca_params.npz https://storage.googleapis.com/audioset/vggish_pca_params.npz
ENV PYTHONPATH "${PYTHONPATH}:/lib/models/research/audioset/vggish"

######### Jupyter #########
RUN pip3 install jupyter
RUN mkdir /notebooks && chmod a+rwx /notebooks
RUN mkdir /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888
EXPOSE 6006
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"]
