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

if [ ! -d $COVERITY_ANALYZER_DIR ]; then
  echo "cannot find coverity-analyzer"
  exit 1
fi

which $COVERITY_ANALYZER_DIR/bin/cov-analyze
if [[ $? -ne 0 ]]; then
  echo "cannot find coverity-analyzer"
  exit 1
fi

pushd $HOST_HOME

# prepare rootfs
if [[ ! -d $ROOTFS_DIR ]]; then
  echo "cannot find rootfs"
  exit 1
fi

DOCKER_VOLUMES+=" -v $COVERITY_ANALYZER_DIR:/opt/cov-analyzer"
DOCKER_VOLUMES+=" -v $ROOTFS_DIR:/opt/rootfs"
DOCKER_VOLUMES+=" -v $COVERITY_OUT:/opt/cov-out"

if [ -n "$DOCKER_INTERACTIVE" ]; then
  DOCKER_RUN_OPTS+=" -it"
  CMD="/bin/bash"
else
  CMD="make acl tflite && /opt/cov-analyzer/bin/cov-configure --template --compiler arm-linux-gnueabihf-gcc && /opt/cov-analyzer/bin/cov-build --dir /opt/cov-out make runtime testbuild"
fi

docker run $DOCKER_RUN_OPTS $DOCKER_ENV_VARS $DOCKER_VOLUMES $DOCKER_IMAGE_NAME sh -c "$CMD"
BUILD_RESULT=$?

source $SCRIPT_ROOT/../docker_helper
restore_ownership $HOST_HOME $DOCKER_HOME

# Reset host name (configuration of coverity is done in docker container)
$COVERITY_ANALYZER_DIR/bin/cov-manage-emit --dir $COVERITY_OUT reset-host-name

# Analyze build result
$COVERITY_ANALYZER_DIR/bin/cov-analyze \
  --dir $COVERITY_OUT --jobs auto --all --disable-test-metrics --security \
  --concurrency --enable-constraint-fpp --fnptr-models --enable SIZECHECK \
  --checker-option STRING_OVERFLOW:report_fixed_size_dest:no --checker-option OVERRUN:allow_symbol:no \
  --checker-option PASS_BY_VALUE:unmodified_threshold:1024 --checker-option PASS_BY_VALUE:size_threshold:1024 \
  -co NULL_RETURNS:stat_threshold:70 --enable INTEGER_OVERFLOW --enable INCOMPATIBLE_CAST \
  --enable IDENTICAL_BRANCHES --enable UNINTENDED_INTEGER_DIVISION

popd

exit $BUILD_RESULT
