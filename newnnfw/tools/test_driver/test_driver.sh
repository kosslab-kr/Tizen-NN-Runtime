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

set -e
# NOTE: Supposed that this script would be executed with an artifact path.
#       The artifact path has tests/(test suite) and Product/
#       Reference this PR(https://github.sec.samsung.net/STAR/nnfw/pull/375).

function Usage()
{
    echo "Usage: ./$0 --artifactpath=.    # run all tests"
    echo "Usage: ./$0 --artifactpath=/home/dragon/nnfw --frameworktest --verification --benchmark    # run fw test & verfication and benchmark"
    echo ""
    echo "--artifactpath            - (default={test_driver.sh's path}/../../) it should contain tests/ and Product/"
    echo ""
    echo "Following options are needed when you want to tests of specific types. If you don't pass any one, unittest and verification will be run"
    echo "--unittest                - (default=on) run unit test"
    echo "--unittestall             - (default=off) run all unit test without skip, overrite --unittest option"
    echo "--frameworktest           - (default=off) run framework test"
    echo "--verification            - (default=on) run verification"
    echo "--frameworktest_list_file - filepath of model list for test"
    echo ""
    echo "Following option is only needed when you want to test benchmark."
    echo "--benchmark_acl           - (default=off) run benchmark-acl"
    echo "--benchmark               - (default=off) run benchmark"
    echo "--benchmark_op            - (default=off) run benchmark per operation"
    echo "--benchmark_tflite_model  - (default=off) run tflite_benchmark_model"
    echo ""
    echo "Following option is used for profiling."
    echo "--profile                 - (default=off) run operf"
    echo ""
    echo "etc."
    echo "--framework_driverbin     - (default=../../Product/out/bin/tflite_run) runner for runnning framework tests"
    echo "--verification_driverbin  - (default=../../Product/out/bin/nnapi_test) runner for runnning verification tests"
    echo "--benchmark_driverbin     - (default=../../Product/out/bin/tflite_benchmark) runner for runnning benchmark"
    echo "--runtestsh               - (default=\$ARTIFACT_PATH/tests/framework/run_test.sh) run_test.sh with path where it is for framework test and verification"
    echo "--unittestdir             - (default=\$ARTIFACT_PATH/Product/out/unittest) directory that has unittest binaries for unit test"
    echo ""
    echo "--ldlibrarypath           - (default=\$ARTIFACT_PATH/Product/out/lib) path that you want to include libraries"
    echo "--usennapi                - (default=on)  declare USE_NNAPI=1"
    echo "--nousennapi              - (default=off) declare nothing about USE_NNAPI"
    echo "--acl_envon               - (default=off) declare envs for ACL"
    echo "--reportdir               - (default=\$ARTIFACT_PATH/report) directory to save report"
    echo ""
}

TEST_DRIVER_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
ARTIFACT_PATH="$TEST_DRIVER_DIR/../../"
FRAMEWORK_DRIVER_BIN=""
VERIFICATION_DRIVER_BIN=""
BENCHMARK_DRIVER_BIN=""
RUN_TEST_SH=""
UNIT_TEST_DIR=""
LD_LIBRARY_PATH_IN_SHELL=""
USE_NNAPI="USE_NNAPI=1"
ALLTEST_ON="true"
UNITTEST_ON="false"
UNITTESTALL_ON="false"
FRAMEWORKTEST_ON="false"
VERIFICATION_ON="false"
BENCHMARK_ON="false"
BENCHMARK_OP_ON="false"
BENCHMARK_TFLITE_MODEL_ON="false"
BENCHMARK_ACL_ON="false"
ACL_ENV_ON="false"
PROFILE_ON="false"
REPORT_DIR=""

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --artifactpath=*)
            ARTIFACT_PATH=${i#*=}
            ;;
        --framework_driverbin=*)
            FRAMEWORK_DRIVER_BIN=${i#*=}
            ;;
        --verification_driverbin=*)
            VERIFICATION_DRIVER_BIN=${i#*=}
            ;;
        --benchmark_driverbin=*)
            BENCHMARK_DRIVER_BIN=${i#*=}
            ;;
        --runtestsh=*)
            RUN_TEST_SH=${i#*=}
            ;;
        --unittestdir=*)
            UNIT_TEST_DIR=${i#*=}
            ;;
        --ldlibrarypath=*)
            LD_LIBRARY_PATH_IN_SHELL=${i#*=}
            ;;
        --usennapi)
            USE_NNAPI="USE_NNAPI=1"
            ;;
        --nousennapi)
            USE_NNAPI=""
            ;;
        --unittest)
            ALLTEST_ON="false"
            UNITTEST_ON="true"
            ;;
        --unittestall)
            ALLTEST_ON="false"
            UNITTEST_ON="true"
            UNITTESTALL_ON="true"
            ;;
        --frameworktest)
            ALLTEST_ON="false"
            FRAMEWORKTEST_ON="true"
            ;;
        --frameworktest_list_file=*)
            FRAMEWORKTEST_LIST_FILE=$PWD/${i#*=}
            if [ ! -e "$FRAMEWORKTEST_LIST_FILE" ]; then
                echo "Pass on with proper frameworktest_list_file"
                exit 1
            fi
            ;;
        --verification)
            ALLTEST_ON="false"
            VERIFICATION_ON="true"
            ;;
        --benchmark)
            ALLTEST_ON="false"
            BENCHMARK_ON="true"
            ;;
        --benchmark_op)
            ALLTEST_ON="false"
            BENCHMARK_OP_ON="true"
            ;;
        --benchmark_tflite_model)
            ALLTEST_ON="false"
            BENCHMARK_TFLITE_MODEL_ON="true"
            ;;
        --benchmark_acl)
            ALLTEST_ON="false"
            BENCHMARK_ACL_ON="true"
            ;;
        --acl_envon)
            ACL_ENV_ON="true"
            ;;
        --profile)
            ALLTEST_ON="false"
            PROFILE_ON="true"
            ;;
        --reportdir=*)
            REPORT_DIR=${i#*=}
            ;;
        *)
            # Be careful that others params are handled as $ARTIFACT_PATH
            ARTIFACT_PATH="$i"
            ;;
    esac
    shift
done

ARTIFACT_PATH="$(readlink -f $ARTIFACT_PATH)"

if [ -z "$RUN_TEST_SH" ]; then
    RUN_TEST_SH=$ARTIFACT_PATH/tests/framework/run_test.sh
fi

if [ ! -e "$RUN_TEST_SH" ]; then
    echo "Cannot find $RUN_TEST_SH"
    exit 1
fi

if [ -z "$UNIT_TEST_DIR" ]; then
    UNIT_TEST_DIR=$ARTIFACT_PATH/Product/out/unittest
fi

if [ -z "$REPORT_DIR" ]; then
    REPORT_DIR=$ARTIFACT_PATH/report
fi

if [ -z "$LD_LIBRARY_PATH_IN_SHELL" ]; then
    LD_LIBRARY_PATH="$ARTIFACT_PATH/Product/out/lib:$LD_LIBRARY_PATH"
else
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH_IN_SHELL:$LD_LIBRARY_PATH"
fi

# Set env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
if [ -n "$USE_NNAPI" ]; then
    export "$USE_NNAPI"
fi

source $TEST_DRIVER_DIR/common.sh

if [ "$ACL_ENV_ON" == "true" ]; then
    switch_nnfw_kernel_env "ON" "acl"
fi

# Run unittest in each part such as Runtime, ACL
if [ "$ALLTEST_ON" == "true" ] || [ "$UNITTEST_ON" == "true" ]; then
    if [ "$UNITTESTALL_ON" == "true" ]; then
        $TEST_DRIVER_DIR/run_unittest.sh \
            --reportdir=$REPORT_DIR \
            --unittestdir=$UNIT_TEST_DIR \
            --runall
    else
        $TEST_DRIVER_DIR/run_unittest.sh \
            --reportdir=$REPORT_DIR \
            --unittestdir=$UNIT_TEST_DIR
    fi
fi

# Run tflite_run with various tflite models
if [ "$FRAMEWORKTEST_ON" == "true" ]; then
    if [ -z "$FRAMEWORK_DRIVER_BIN" ]; then
        FRAMEWORK_DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_run
    fi

    $TEST_DRIVER_DIR/run_frameworktest.sh \
        --runtestsh=$RUN_TEST_SH \
        --driverbin=$FRAMEWORK_DRIVER_BIN \
        --reportdir=$REPORT_DIR \
        --tapname=framework_test.tap \
        --logname=framework_test.log \
        --testname="Frameworktest" \
        --frameworktest_list_file=${FRAMEWORKTEST_LIST_FILE:-}
fi

# Run nnapi_test with various tflite models
if [ "$ALLTEST_ON" == "true" ] || [ "$VERIFICATION_ON" == "true" ]; then
    if [ -z "$VERIFICATION_DRIVER_BIN" ]; then
        VERIFICATION_DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/nnapi_test
    fi

    # verification uses the same script as frameworktest does
    $TEST_DRIVER_DIR/run_frameworktest.sh \
        --runtestsh=$RUN_TEST_SH \
        --driverbin=$VERIFICATION_DRIVER_BIN \
        --reportdir=$REPORT_DIR \
        --tapname=verification_test.tap \
        --logname=verification_test.log \
        --testname="Verification" \
        --frameworktest_list_file=${FRAMEWORKTEST_LIST_FILE:-}
fi

# Run tflite_benchmark with tflite models
if [ "$BENCHMARK_ON" == "true" ]; then
    if [ -z "$BENCHMARK_DRIVER_BIN" ]; then
        DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_benchmark
    else
        DRIVER_BIN=$BENCHMARK_DRIVER_BIN
    fi

    $TEST_DRIVER_DIR/run_benchmark.sh \
        --runtestsh=$RUN_TEST_SH \
        --driverbin=$DRIVER_BIN \
        --reportdir=$REPORT_DIR/benchmark
fi

# Run tflite_benchmark from a list of tflite models.
# Each model has only one operator.
if [ "$BENCHMARK_OP_ON" == "true" ]; then
    if [ -z "$BENCHMARK_DRIVER_BIN" ]; then
        DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_benchmark
    else
        DRIVER_BIN=$BENCHMARK_DRIVER_BIN
    fi

    $TEST_DRIVER_DIR/run_benchmark_op.sh \
        --runtestsh=$RUN_TEST_SH \
        --driverbin=$DRIVER_BIN \
        --reportdir=$REPORT_DIR/benchmark_op \
        --modelfilepath=$ARTIFACT_PATH/tests/framework \
        --frameworktest_list_file=${FRAMEWORKTEST_LIST_FILE:-}
fi

# Run benchmark/acl/benchmark_googlenet, mobilenet and inception_v3
if [ "$BENCHMARK_ACL_ON" == "true" ]; then
    $TEST_DRIVER_DIR/run_benchmark_acl.sh \
        --reportdir=$AREPORT_DIR/benchmark \
        --bindir=$ARTIFACT_PATH/Product/out/bin
fi

# Make json file. Actually, this process is only needed on CI. That's why it is in test_driver.sh.
if [ "$BENCHMARK_ON" == "true" ] || [ "$BENCHMARK_ACL_ON" == "true" ] || [ "$BENCHMARK_OP_ON" == "true" ]; then
    # functions to fill json with benchmark results
    source $ARTIFACT_PATH/tools/test_driver/print_to_json.sh
    if [ "$BENCHMARK_OP_ON" == "true" ]; then
        print_to_json $REPORT_DIR/benchmark_op $REPORT_DIR "benchmark_op_result.json"
    else
        print_to_json $REPORT_DIR/benchmark $REPORT_DIR "benchmark_result.json"
    fi
fi

# Run tflite_benchmark_model (= per-operation profiling tool).
# Each model can contain arbitrary number of operators.
if [ "$BENCHMARK_TFLITE_MODEL_ON" == "true" ]; then
    $TEST_DRIVER_DIR/run_benchmark_tflite_model.sh \
        --reportdir=$REPORT_DIR/benchmark_tflite_model \
        --modelroot=$ARTIFACT_PATH/tests/framework/tests
fi

# Run profiling
if [ "$PROFILE_ON" == "true" ]; then
    # FIXME: These driver and tflite test are set temporarily. Fix these to support flexibility
    DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_run
    TFLITE_TEST=$ARTIFACT_PATH/tests/framework/cache/inceptionv3/inception_module/inception_test.tflite

    # TODO: Enable operf to set directory where sample data puts on
    rm -rf oprofile_data

    echo ""
    echo "============================================"
    operf -g $DRIVER_BIN $TFLITE_TEST
    echo "============================================"
    echo ""
fi
