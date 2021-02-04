# もととなるImage 
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# 走らせるbashコマンド
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install vim wget curl
# RUN apt-get install -y libsm6 libxext6 libxrender-dev
# RUN apt-get -y install libopencv-dev
# RUN apt-get -y install jupyter-notebook

RUN pip3 install -U pip
RUN pip3 install jupyter numpy matplotlib seaborn pandas tqdm
RUN pip3 install torch torchvision
RUN pip3 install torchsummary
# RUN pip3 install opencv-python

RUN pip3 install streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# duser setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
    adduser duser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER duser
# 各種命令を実行するカレントディレクトリを指定
WORKDIR /home/duser/