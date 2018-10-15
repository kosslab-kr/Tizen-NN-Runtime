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

BENCHMARK_RUN_TEST_SH=
BENCHMARK_DRIVER_BIN=
BENCHMARK_REPORT_DIR=
BENCHMARK_MODELS_FILE=
BENCHMARK_MODEL_LIST="inceptionv3/inception_nonslim inceptionv3/inception_slim mobilenet"

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
        --runtestsh=*)
            BENCHMARK_RUN_TEST_SH=${i#*=}
            ;;
        --driverbin=*)
            BENCHMARK_DRIVER_BIN=${i#*=}
            ;;
        --reportdir=*)
            BENCHMARK_REPORT_DIR=${i#*=}
            BENCHMARK_MODELS_FILE=$BENCHMARK_REPORT_DIR/benchmark_models.txt
            ;;
    esac
    shift
done

function get_result_of_benchmark_test()
{
    local RUN_TEST_SH=$1
    local DRIVER_BIN=$2
    local MODEL=$3
    local LOG_FILE=$4

    local RET=0
    $RUN_TEST_SH --driverbin=$DRIVER_BIN $MODEL > $LOG_FILE 2>&1
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "Testing $MODEL aborted... exit code: $RET"
        exit $RET
    fi

    local RESULT=`grep -E '^Mean:' $LOG_FILE | sed -e 's/ms//g' | awk '{print $2}'`
    echo "$RESULT"
}

function print_result_of_benchmark_test()
{
    local NAME=$1
    local RESULT=$2
    local RESULT_FILE=$3

    echo "$NAME $RESULT" > $RESULT_FILE
}

function run_benchmark_test()
{
    local DRIVER_BIN=$BENCHMARK_DRIVER_BIN
    local LOG_FILE=
    local RESULT_FILE=
    local RESULT=
    local REPORT_MODEL_DIR=

    export COUNT=5
    echo "============================================"
    local i=0
    for MODEL in $BENCHMARK_MODEL_LIST; do
        echo "Benchmark test with `basename $DRIVER_BIN` & `echo $MODEL`"
        echo $MODEL >> $BENCHMARK_MODELS_FILE

        REPORT_MODEL_DIR=$BENCHMARK_REPORT_DIR/$MODEL
        mkdir -p $REPORT_MODEL_DIR

        # TFLite+CPU
        LOG_FILE=$REPORT_MODEL_DIR/tflite_cpu.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_cpu.result
        echo -n "TFLite + CPU................... "
        unset USE_NNAPI
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_CPU" $RESULT $RESULT_FILE

        # TFLite+NNAPI(CPU fallback)
        LOG_FILE=$REPORT_MODEL_DIR/tflite_nnapi_cpu.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_nnapi_cpu.result
        echo -n "TFLite + NNAPI(CPU)............ "
        export USE_NNAPI=1
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_NNAPI_CPU" $RESULT $RESULT_FILE

        # TFLite+NNAPI(ACL)
        LOG_FILE=$REPORT_MODEL_DIR/tflite_nnapi_acl.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_nnapi_acl.result
        echo -n "TFLite + NNAPI(ACL)............ "
        switch_nnfw_kernel_env "ON" "acl"
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_NNAPI_ACL" $RESULT $RESULT_FILE
        unset USE_NNAPI
        switch_nnfw_kernel_env "OFF"

        if [[ $i -ne $(echo $BENCHMARK_MODEL_LIST | wc -w)-1 ]]; then
            echo ""
        fi
        i=$((i+1))
    done
    echo "============================================"
    unset COUNT
}

if [ ! -e "$BENCHMARK_REPORT_DIR" ]; then
    mkdir -p $BENCHMARK_REPORT_DIR
fi

rm -rf $BENCHMARK_MODELS_FILE

echo ""
run_benchmark_test
echo ""
