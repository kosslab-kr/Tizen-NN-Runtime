#!/bin/bash

# _COVERITY_USER,  _COVERITY_PASSWORD, _COVERITY_STREAM and COVERITY_SCRIPT_DOWNLOAD_URL should be defined.
# XXX are used only in this script and  _XXX are used in Coverity script.
if [ -z ${_COVERITY_STREAM+x} ]; then
    echo "_COVERITY_STREAM is unset";
    exit 1
else
    echo "_COVERITY_STREAM is set to '$_COVERITY_STREAM'";
fi

if [ -z ${_COVERITY_USER+x} ]; then
    echo "_COVERITY_USER is unset";
    exit 1
else
    echo "_COVERITY_USER is set to '$_COVERITY_USER'";
fi

if [ -z ${_COVERITY_PASSWORD+x} ]; then
    echo "_COVERITY_PASSWORD is unset";
    exit 1
else
    echo "_COVERITY_PASSWORD is set to '$_COVERITY_PASSWORD'";
fi

if [ -z ${COVERITY_SCRIPT_DOWNLOAD_URL+x} ]; then
    echo "COVERITY_SCRIPT_DOWNLOAD_URL is unset";
    exit 1
else
    echo "COVERITY_SCRIPT_DOWNLOAD_URL is set to '$COVERITY_SCRIPT_DOWNLOAD_URL'";
fi

#
# Set variables required for Coverity script
#

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOST_HOME=$(readlink -f $SCRIPT_ROOT/../..)

export _BUILD_PATH=$HOST_HOME
export _RUN_COVERITY_ROOT=$_BUILD_PATH/ci_coverity
mkdir -p $_RUN_COVERITY_ROOT

export GBS_BUILDROOT=$_RUN_COVERITY_ROOT/GBS-ROOT/
mkdir -p $GBS_BUILDROOT

# Prepare gbs.conf for Coverity
sed -- 's/^buildroot = .*/buildroot = '${GBS_BUILDROOT//\//\\/}'/' < $SCRIPT_ROOT/gbs.conf > $SCRIPT_ROOT/gbs_coverity.conf
export _BUILD_CMD="gbs -c ${SCRIPT_ROOT}/gbs_coverity.conf build -A armv7l --profile=profile.tizen --clean-repo"
export _PROD_LOCATION=$GBS_BUILDROOT/local/repos/tizen/armv7l/RPMS

export _COVERITY_BINARY_DIR=$_RUN_COVERITY_ROOT/coverity-binary
mkdir -p $_COVERITY_BINARY_DIR

# Invoke Coverity script
pushd $_RUN_COVERITY_ROOT
wget $COVERITY_SCRIPT_DOWNLOAD_URL
popd
chmod +x $_RUN_COVERITY_ROOT/${COVERITY_SCRIPT_DOWNLOAD_URL##*/}
$_RUN_COVERITY_ROOT/${COVERITY_SCRIPT_DOWNLOAD_URL##*/}
