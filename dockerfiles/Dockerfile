# Set the base image to Ubuntu
FROM tensorflow/tensorflow:latest-py3

# File Author / Maintainer
MAINTAINER Raul Puri

# Install git and TF dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libboost-all-dev && \
    apt-get install -y software-properties-common \
    git \
    wget \
    cmake \
    python-zmq \
    python-dev \
    libzmq3-dev \
    libssl-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    protobuf-compiler

COPY requirements.txt /root/

RUN pip install keras nose Cython
RUN pip install -r /root/requirements.txt
RUN rm /root/requirements.txt


WORKDIR /root

CMD ["/bin/bash"]
