#!/bin/bash

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

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH=$SCRIPT_PATH/../..
FLATC=$ROOT_PATH/Product/out/bin/flatc

if [ ! -e "$1" ]; then
  echo "file not exists: $1"
  exit 1
fi

JSON_FILE=$1
JSON_FILENAME=${TFLITE_FILE##*\/}
TFLITE_FILENAME=${TFLITE_FILENAME%\.json}.tflite

$FLATC -b $ROOT_PATH/externals/tensorflow/tensorflow/contrib/lite/schema/schema.fbs $JSON_FILE

