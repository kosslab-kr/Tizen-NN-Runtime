#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DOCKER_HOME and DOCKER_ENV_VARS
source $SCRIPT_ROOT/nnfw_docker

HOST_HOME=$SCRIPT_ROOT/../..

DOCKER_VOLUMES+=" -v $HOST_HOME:$DOCKER_HOME"

DOCKER_ENV_VARS+=" -e TARGET_ARCH=armv7l"
DOCKER_ENV_VARS+=" -e CROSS_BUILD=1"
DOCKER_ENV_VARS+=" -e ROOTFS_DIR=/opt/rootfs"
DOCKER_ENV_VARS+=" -e EXT_ACL_FOLDER=/opt/libarmcl"

DOCKER_RUN_OPTS="--rm"
DOCKER_RUN_OPTS+=" -w $DOCKER_HOME"

TMP_DIR=$HOST_HOME/tmp

if [ ! -d $SVACE_ANALYZER_DIR ]; then
  echo "cannot find svace-analyzer"
  exit 1
fi

which $SVACE_ANALYZER_DIR/bin/svace
if [[ $? -ne 0 ]]; then
  echo "cannot find svace-analyzer"
  exit 1
fi

pushd $HOST_HOME

# prepare rootfs
if [[ ! -d $ROOTFS_DIR ]]; then
  echo "cannot find rootfs"
  exit 1
fi

# prepare svace
if [[ ! -f $SVACE_POLICY_FILE ]]; then
  echo "cannot find svace policy"
  exit 1
fi

DOCKER_VOLUMES+=" -v $SVACE_ANALYZER_DIR:/opt/svace-analyzer"
DOCKER_VOLUMES+=" -v $ROOTFS_DIR:/opt/rootfs"

if [ -n "$DOCKER_INTERACTIVE" ]; then
  DOCKER_RUN_OPTS+=" -it"
  CMD="/bin/bash"
else
  CMD="make acl tflite && /opt/svace-analyzer/bin/svace init && /opt/svace-analyzer/bin/svace build make runtime testbuild"
fi

docker run $DOCKER_RUN_OPTS $DOCKER_ENV_VARS $DOCKER_VOLUMES $DOCKER_IMAGE_NAME sh -c "$CMD"
BUILD_RESULT=$?

source $SCRIPT_ROOT/../docker_helper
restore_ownership $HOST_HOME $DOCKER_HOME

$SVACE_ANALYZER_DIR/bin/svace analyze --warning $SVACE_POLICY_FILE

popd

exit $BUILD_RESULT
