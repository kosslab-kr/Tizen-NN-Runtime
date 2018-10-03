#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DOCKER_HOME and DOCKER_ENV_VARS
source $SCRIPT_ROOT/nnfw_docker

HOST_HOME=$SCRIPT_ROOT/../..

DOCKER_VOLUMES+=" -v $HOST_HOME:$DOCKER_HOME"

DOCKER_ENV_VARS+=" -e TARGET_ARCH=armv7l"
DOCKER_ENV_VARS+=" -e CROSS_BUILD=1"
DOCKER_ENV_VARS+=" -e ROOTFS_DIR=/opt/rootfs"

DOCKER_RUN_OPTS="--rm"
DOCKER_RUN_OPTS+=" -w $DOCKER_HOME"

# prepare rootfs
if [[ ! -d $ROOTFS_DIR ]]; then
    echo "cannot find rootfs"
    exit 1
fi

if [[ ! -z $ENV_FILE ]]; then
    DOCKER_ENV_VARS+=" --env-file ${ENV_FILE} "
fi

DOCKER_VOLUMES+=" -v $ROOTFS_DIR:/opt/rootfs"

if [ -n "$DOCKER_INTERACTIVE" ]; then
  DOCKER_RUN_OPTS+=" -it"
  CMD="/bin/bash"
else
  CMD="export BENCHMARK_ACL_BUILD=1 && make acl && make && make install && make build_test_suite"
fi

docker run $DOCKER_RUN_OPTS $DOCKER_ENV_VARS $DOCKER_VOLUMES $DOCKER_IMAGE_NAME sh -c "$CMD"
BUILD_RESULT=$?

source $SCRIPT_ROOT/../docker_helper
restore_ownership $HOST_HOME $DOCKER_HOME

exit $BUILD_RESULT
