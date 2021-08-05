FROM ubuntu:18.04
MAINTAINER Jihun Hong <hjihun0643@gmail.com>

RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip
RUN python3.6 -m pip install pip --upgrade

RUN python3.6 -m pip install tensorflow==2.4.1
RUN python3.6 -m pip install tensorflow_addons
RUN python3.6 -m pip install opencv-python
RUN python3.6 -m pip install sklearn
#RUN sudo apt-get install -y libgl1-mesa-dev
#RUN sudo apt-get install -y libgl1-mesa-glx
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

WORKDIR ./alexNet/
#VOLUME /root/home/hong/PycharmProjects/pythonProject/alexNet