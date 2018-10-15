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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NNFW_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
REPORT_DIR="$NNFW_DIR/report/tflite_benchmark_model"
MODEL_ROOT="$NNFW_DIR/tests/framework/tests"
LD_LIBRARY_PATH="$NNFW_DIR/Product/out/lib"

RUN_TEST=$NNFW_DIR/tests/framework/run_test.sh
MODEL_IN=${BASH_SOURCE[0]%.sh}.in
BENCHMARK_BIN=$NNFW_DIR/Product/out/bin/tflite_benchmark_model
MODEL_NAMES=
MODEL_PARAMS=

source $SCRIPT_DIR/common.sh

usage()
{
  echo
  echo "Usage: LD_LIBRARY_PATH=Product/out/lib $(basename ${BASH_SOURCE[0]}) --reportdir=report --modelroot=modelroot"
  echo
}

parse_args()
{
  for i in "$@"; do
    case $i in
      -h|--help|help)
        usage
        exit 1
        ;;
      --reportdir=*)
        REPORT_DIR=${i#*=}
        ;;
      --modelroot=*)
        MODEL_ROOT=${i#*=}
        ;;
    esac
    shift
  done
}

load_input()
{
  mapfile -t MODEL_NAMES  < <(cut -f1  "${MODEL_IN}")
  mapfile -t MODEL_PARAMS < <(cut -f2- "${MODEL_IN}")
  if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No model is found. Please check ${MODEL_IN} is correct."
    exit 1
  fi
}

download_models()
{
  $RUN_TEST --download=on $MODEL_NAMES
}

run_benchmarks()
{
  echo
  echo "Running benchmarks:"
  echo "======================"

  for (( i=0; i< ${#MODEL_NAMES[@]}; i++)); do
    MODEL_NAME=${MODEL_NAMES[i]}
    MODEL_PATH=$(find $NNFW_DIR/tests/framework/cache/$MODEL_NAME/ -name "*.tflite")
    MODEL_PARAM=${MODEL_PARAMS[$i]}

    echo "$MODEL_NAME"

    local REPORT_MODEL_DIR=$REPORT_DIR/$MODEL_NAME
    mkdir -p $REPORT_MODEL_DIR

    local OUT_FILE

    # TFLite Interpreter
    OUT_FILE=$REPORT_MODEL_DIR/tflite_interpreter.out
    echo
    echo "{ // TFLite Interpreter"
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BENCHMARK_BIN --graph=$MODEL_PATH $MODEL_PARAM --use_nnapi=false 2> >(tee $OUT_FILE)
    echo "} // TFLite Interpreter"

    # TFLite PureACL (CL)
    OUT_FILE=$REPORT_MODEL_DIR/tflite_pureacl_cl.out
    echo
    echo "{ // TFLite PureACL(CL)"
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BENCHMARK_BIN --graph=$MODEL_PATH $MODEL_PARAM --use_nnapi=true 2> >(tee $OUT_FILE)
    echo "} // TFLite_PureACL(CL)"
  done
}

# for debug
print_vars()
{
  echo SCRIPT_DIR=$SCRIPT_DIR
  echo NNFW_DIR=$NNFW_DIR
  echo RUN_TEST=$RUN_TEST
  echo MODEL_IN=$MODEL_IN
  echo BENCHMARK_BIN=$BENCHMARK_BIN
  echo REPORT_DIR=$REPORT_DIR
  echo MODEL_ROOT=$MODEL_ROOT
}

if [ ! -e "$REPORT_DIR" ]; then
    mkdir -p $REPORT_DIR
fi

parse_args $@
load_input
download_models
run_benchmarks
