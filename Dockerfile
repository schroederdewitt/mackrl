FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER ANONYMOUS

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*


#Install python3 pip3
RUN apt-get update
RUN apt-get update && apt-get -y install python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger

RUN mkdir /install
WORKDIR /install

#### -------------------------------------------------------------------
#### install MongoDB (for Sacred)
#### -------------------------------------------------------------------

# Install pymongo
RUN pip3 install pymongo

#### -------------------------------------------------------------------
#### install pysc2 #(from Mika fork)
#### -------------------------------------------------------------------

# RUN git clone https://github.com/samvelyan/pysc2.git /install/pysc2
#RUN git clone https://github.com/deepmind/pysc2.git /install/pysc2 && cd /install/pysc2 && git checkout 65f8badf1b3cbc0d711a8d4c87e4501225b1c0fa
#RUN pip3 install /install/pysc2/
#RUN pip3 install /install/pysc2/
RUN git clone https://github.com/deepmind/pysc2.git /install/pysc2
RUN pip3 install /install/pysc2/

#### -------------------------------------------------------------------
#### install Sacred
#### -------------------------------------------------------------------

RUN pip3 install setuptools
RUN git clone https://github.com/idsia/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### final steps
#### -------------------------------------------------------------------
RUN pip3 install pygame

#### -------------------------------------------------------------------
#### Plotting tools
#### -------------------------------------------------------------------

# RUN apt-get -y install ipython ipython-notebook
RUN pip3 install statsmodels pandas seaborn
RUN mkdir /pool && echo "export PATH=$PATH:'/pool/pool'" >> ~/.bashrc

RUN pip3 install cloudpickle ruamel.yaml

RUN apt-get install -y libhdf5-serial-dev cmake
RUN git clone https://github.com/Blosc/c-blosc.git /install/c-blosc && cd /install/c-blosc && cmake -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --build . --target install
RUN pip3 install tables h5py

#### -------------------------------------------------------------------
#### install tensorflow
#### -------------------------------------------------------------------
RUN pip3 install tensorflow-gpu
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp34-cp34m-linux_x86_64.whl

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------

# RUN git clone https://github.com/csarofeen/pytorch /install/pytorch && cd /install/pytorch 
# RUN pip3 install numpy pyyaml mkl setuptools cffi
# RUN apt-get install -y cmake gcc 
# RUN cd /install/pytorch && python3 setup.py install
#RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install torch
#RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision snakeviz pytest probscale
RUN apt-get install -y htop iotop

EXPOSE 8888

WORKDIR /fastmarl
# RUN echo "mongod --fork --logpath /var/log/mongod.log" >> ~/.bashrc
#CMD ["mongod", "--fork", "--logpath", "/var/log/mongod.log"]
# EXPOSE 27017
# EXPOSE 28017

# CMD service mongod start && tail -F /dev/null
