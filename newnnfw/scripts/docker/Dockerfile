FROM ubuntu:16.04

ENV http_proxy $http_proxy
ENV https_proxy $https_proxy

RUN apt-get update && apt-get --yes --force-yes install build-essential
RUN apt-get update && apt-get --yes --force-yes install scons cmake
RUN apt-get update && apt-get --yes --force-yes install libboost-all-dev
RUN apt-get update && apt-get --yes --force-yes install git
RUN apt-get update && apt-get --yes --force-yes install gcc-5-arm-linux-gnueabi g++-5-arm-linux-gnueabi
RUN apt-get update && apt-get --yes --force-yes install lcov
RUN apt-get update && apt-get --yes --force-yes install clang-format-3.9 python-pip
RUN pip install yapf
RUN apt-get update && apt-get --yes --force-yes install doxygen graphviz
RUN apt-get update && apt-get --yes --force-yes install wget
RUN wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.02/arm-linux-gnueabihf/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf.tar.xz -O gcc.tar.xz -nv
RUN tar -xf gcc.tar.xz -C /opt/ && rm -rf gcc.tar.xz
ENV PATH "/opt/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/bin:$PATH"
