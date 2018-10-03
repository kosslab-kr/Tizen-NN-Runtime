#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DOCKER_HOME and DOCKER_ENV_VARS
source $SCRIPT_ROOT/nnfw_docker

HOST_HOME=$SCRIPT_ROOT/../..

DOCKER_VOLUMES+=" -v $HOST_HOME:$DOCKER_HOME"

DOCKER_RUN_OPTS="--rm"
DOCKER_RUN_OPTS+=" -w $DOCKER_HOME"
DOCKER_ENV_VARS+=" -e DOCKER_HOME=${DOCKER_HOME}"

docker run $DOCKER_RUN_OPTS $DOCKER_ENV_VARS $DOCKER_VOLUMES $DOCKER_IMAGE_NAME sh -c "./scripts/command/gen_coverage_report.sh"

BUILD_RESULT=$?

source $SCRIPT_ROOT/../docker_helper
restore_ownership $HOST_HOME $DOCKER_HOME

exit $BUILD_RESULT

