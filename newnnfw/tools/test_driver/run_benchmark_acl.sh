#!/bin/bash
#
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $MY_PATH/common.sh

BENCHMARKACL_BIN_DIR=
BENCHMARKACL_REPORT_DIR=
BENCHMARKACL_MODELS_FILE=
BENCHMARKACL_MODEL_LIST="inceptionv3/inception_nonslim inceptionv3/inception_slim"

function Usage()
{
    # TODO: Fill this
    echo "Usage: LD_LIBRARY_PATH=Product/out/lib ./$0 --reportdir=report"
}

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --reportdir=*)
            BENCHMARKACL_REPORT_DIR=${i#*=}
            BENCHMARKACL_MODELS_FILE=$BENCHMARKACL_REPORT_DIR/benchmarkacl_models.txt
            ;;
        --bindir=*)
            BENCHMARKACL_BIN_DIR=${i#*=}
            ;;
    esac
    shift
done

function run_benchmark_acl()
{
    local REPORT_DIR=$BENCHMARKACL_REPORT_DIR
    local DRIVER_DIR=$BENCHMARKACL_BIN_DIR
    local LOG_FILE=""
    local RESULT_FILE=""
    local RESULT=""
    local RET=0

    export COUNT=5
    echo "============================================"
    local i=0
    for BENCHMARK_ACL_BIN in $(ls $DRIVER_DIR/benchmark_*); do
        local BENCHMARK_ACL_BIN_BASENAME=$(basename $BENCHMARK_ACL_BIN)
        mkdir -p $REPORT_DIR/$BENCHMARK_ACL_BIN_BASENAME
        echo "Benchmark/acl test by $BENCHMARK_ACL_BIN_BASENAME"
        echo $BENCHMARK_ACL_BIN_BASENAME >> $BENCHMARKACL_MODELS_FILE

        # ACL(NEON)
        LOG_FILE=$REPORT_DIR/$BENCHMARK_ACL_BIN_BASENAME/acl_neon.txt
        RESULT_FILE=$REPORT_DIR/$BENCHMARK_ACL_BIN_BASENAME/acl_neon.result
        echo -n "ACL(NEON)...... "
        $BENCHMARK_ACL_BIN 0 > $LOG_FILE 2>&1
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "aborted... exit code: $RET"
            exit $RET
        fi
        RESULT=`grep -E '^Mean:' $LOG_FILE | sed -e 's/ms//g' | awk '{print $2}'`
        echo "$RESULT ms"
        echo "ACL(NEON)" $RESULT > $RESULT_FILE

        # ACL(OpenCL)
        LOG_FILE=$REPORT_DIR/$BENCHMARK_ACL_BIN_BASENAME/acl_opencl.txt
        RESULT_FILE=$REPORT_DIR/$BENCHMARK_ACL_BIN_BASENAME/acl_opencl.result
        echo -n "ACL(OpenCL).... "
        $BENCHMARK_ACL_BIN 1 > $LOG_FILE 2>&1
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "aborted... exit code: $RET"
            exit $RET
        fi
        RESULT=`grep -E '^Mean:' $LOG_FILE | sed -e 's/ms//g' | awk '{print $2}'`
        echo "$RESULT ms"
        echo "ACL(OpenCL)" $RESULT > $RESULT_FILE

        if [[ $i -ne $(ls $DRIVER_DIR/benchmark_* | wc -w)-1 ]]; then
            echo ""
        fi
        i=$((i+1))
    done
    echo "============================================"
    unset COUNT
}

if [ ! -e "$BENCHMARKACL_REPORT_DIR" ]; then
    mkdir -p $BENCHMARKACL_REPORT_DIR
fi

rm -rf $BENCHMARKACL_MODELS_FILE

echo ""
run_benchmark_acl
echo ""
