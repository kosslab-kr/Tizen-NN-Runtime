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
NNFW_HOME="$(dirname $(dirname ${MY_PATH}))"
source $MY_PATH/common.sh

BENCHMARK_RUN_TEST_SH=
BENCHMARK_DRIVER_BIN=
BENCHMARK_REPORT_DIR=
BENCHMARK_MODELS_FILE=
BENCHMARK_MODEL_LIST=
MODEL_CACHE_ROOT_PATH=
MODEL_TEST_ROOT_PATH=
PURE_ACL_RT_LIB_PATH=
PURE_LD_LIBRARY_PATH=
ORIGIN_LD_LIBRARY_PATH=
PURE_ACL_RT_ENV_FILE=$MY_PATH/benchmark_op_list.txt

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
            BENCHMARK_MODELS_FILE=$BENCHMARK_REPORT_DIR/benchmark_op_models.txt
            ;;
        --modelfilepath=*)
            TEST_LIST_PATH=${i#*=}
            MODEL_CACHE_ROOT_PATH=$TEST_LIST_PATH/cache
            MODEL_TEST_ROOT_PATH=$TEST_LIST_PATH/tests
            ;;
        --frameworktest_list_file=*)
            FRAMEWORKTEST_LIST_FILE=${i#*=}
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
    local PUREACL_LD_LIBRARY_PATH=$5

    local RET=0
    $RUN_TEST_SH --driverbin=$DRIVER_BIN --ldlibrarypath=$PUREACL_LD_LIBRARY_PATH $MODEL > $LOG_FILE 2>&1
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

function get_benchmark_op_list()
{
    if [ ! -z "$FRAMEWORKTEST_LIST_FILE" ]; then
        BENCHMARK_MODEL_LIST=$(cat "${FRAMEWORKTEST_LIST_FILE}")
    else
        BENCHMARK_MODEL_LIST=$(cat "${PURE_ACL_RT_ENV_FILE}")
    fi
    echo "BENCHMARK_MODEL_LIST=> $BENCHMARK_MODEL_LIST"  
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
        STATUS="enabled"
        source $MODEL_TEST_ROOT_PATH/$MODEL/config.sh

        LOWER_STATUS="$(echo $STATUS | awk '{print tolower($0)}')"
        if [ "$LOWER_STATUS" == "disabled" ]; then
            echo ""
            echo "Skip $MODEL"
            continue
        fi

        echo "Benchmark test with `basename $DRIVER_BIN` & `echo $MODEL`"
        echo $MODEL >> $BENCHMARK_MODELS_FILE

        REPORT_MODEL_DIR=$BENCHMARK_REPORT_DIR/$MODEL
        mkdir -p $REPORT_MODEL_DIR

        # TFLite(CPU fallback)
        LOG_FILE=$REPORT_MODEL_DIR/tflite_cpu_op.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_cpu_op.result
        echo -n "TFLite(CPU fallback)................... "
        unset USE_NNAPI
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_CPU" $RESULT $RESULT_FILE

        # TFLite+NNRuntime(CPU fallback)
        LOG_FILE=$REPORT_MODEL_DIR/tflite_nnrt_cpu_op.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_nnrt_cpu_op.result
        echo -n "TFLite + NNRuntime(CPU fallback)............ "
        export USE_NNAPI=1
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_NNRT_CPU" $RESULT $RESULT_FILE

        # TFLite+NNRuntime+ACL-Neon
        LOG_FILE=$REPORT_MODEL_DIR/tflite_nnrt_acl_neon_op.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_nnrt_acl_neon_op.result
        echo -n "TFLite + NNRuntime + ACL-Neon............ "
        switch_nnfw_kernel_env "ON" "neon"
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_NNRT_ACL-NEON" $RESULT $RESULT_FILE
        switch_nnfw_kernel_env "OFF"

        # TFLite+NNRuntime+ACL-OpenCL
        LOG_FILE=$REPORT_MODEL_DIR/tflite_nnrt_acl_opencl_op.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_nnrt_acl_opencl_op.result
        echo -n "TFLite + NNRuntime + ACL-OpenCL............ "
        switch_nnfw_kernel_env "ON" "acl"
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_NNRT_ACL-OPENCL" $RESULT $RESULT_FILE
        switch_nnfw_kernel_env "OFF"

        # TFLite+PureACLRuntime+ACL-OpenCL
        if [ ! -d "$PURE_ACL_RT_LIB_PATH" ]; then
            echo "Skip $MODEL in Pure ACL Runtime "
            continue
        fi
        LOG_FILE=$REPORT_MODEL_DIR/tflite_pureaclrt_acl_opencl_op.txt
        RESULT_FILE=$REPORT_MODEL_DIR/tflite_pureaclrt_acl_opencl_op.result
        echo -n "TFLite + PureACLRuntime + ACL-OpenCL............ "
        RESULT=$(get_result_of_benchmark_test $BENCHMARK_RUN_TEST_SH $DRIVER_BIN $MODEL $LOG_FILE $PURE_ACL_RT_LIB_PATH)
        echo "$RESULT ms"
        print_result_of_benchmark_test "TFLite_PUREACLRT_ACL-OPENCL" $RESULT $RESULT_FILE
        unset USE_NNAPI

        if [[ $i -ne $(echo $BENCHMARK_MODEL_LIST | wc -w)-1 ]]; then
            echo ""
        fi
        i=$((i+1))
    done
    unset USE_NNAPI
    unset COUNT
    echo "============================================"
}

if [ ! -e "$BENCHMARK_REPORT_DIR" ]; then
    mkdir -p $BENCHMARK_REPORT_DIR
fi

if [ -z "$PURE_ACL_RT_LIB_PATH" ]; then
    PURE_ACL_RT_LIB_PATH="$NNFW_HOME/Product/out/lib/pureacl"
fi

get_benchmark_op_list

rm -rf $BENCHMARK_MODELS_FILE

echo ""
run_benchmark_test
echo ""
