#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DOCKER_HOME and DOCKER_ENV_VARS
source $SCRIPT_ROOT/nnfw_docker_tizen

HOST_HOME=$SCRIPT_ROOT/../..
DOCKER_NNFW_HOME=${DOCKER_HOME}
DOCKER_RPM_HOME=/home/rpm

if [ "${GBS_RPM_DIR}" == "" ];
then
    GBS_RPM_DIR=$HOST_HOME/Product/out/rpm
    mkdir -p ${GBS_RPM_DIR}
fi

if [ -z ${DOCKER_IMAGE} ];
then
    # use default docker image
    DOCKER_IMAGE=nnfw_docker_tizen:latest
fi


DOCKER_VOLUMES+=" -v ${GBS_RPM_DIR}:${DOCKER_RPM_HOME} -v $HOST_HOME:${DOCKER_NNFW_HOME}"
DOCKER_RUN_OPTS+=" --rm"
DOCKER_RUN_OPTS+=" -w ${DOCKER_NNFW_HOME}"

CMD="gbs -c ${DOCKER_NNFW_HOME}/scripts/command/gbs.conf build -A armv7l --profile=profile.tizen --clean --include-all --define '${GBS_DEFINE}' &&
         cp -rf /home/GBS-ROOT/local/repos/tizen/armv7l/RPMS/*.rpm ${DOCKER_RPM_HOME}/."
docker run $DOCKER_RUN_OPTS $DOCKER_VOLUMES ${DOCKER_ENV_VARS:-} ${DOCKER_IMAGE} sh -c "$CMD"
BUILD_RESULT=$?

source $SCRIPT_ROOT/../docker_helper
restore_ownership $HOST_HOME $DOCKER_NNFW_HOME
restore_ownership $HOST_HOME $DOCKER_RPM_HOME

exit $BUILD_RESULT
