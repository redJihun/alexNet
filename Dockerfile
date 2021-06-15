FROM ubuntu:18.04
MAINTAINER Jihun Hong <hjihun0643@gmail.com>

RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip
RUN python3.6 -m pip install pip --upgrade

RUN python3.6 -m pip install tensorflow==2.4.1
RUN python3.6 -m pip install tensorflow_addons

#VOLUME /root/home/hong/PycharmProjects/pythonProject/alexNet