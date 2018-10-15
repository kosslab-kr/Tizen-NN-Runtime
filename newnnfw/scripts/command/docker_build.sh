#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $SCRIPT_ROOT/common.sh

docker build --build-arg http_proxy="$http_proxy" \
  --build-arg https_proxy="$https_proxy" \
  -t $DOCKER_IMAGE_NAME \
  - < $SCRIPT_ROOT/../docker/Dockerfile
