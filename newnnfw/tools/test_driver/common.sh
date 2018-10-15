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

function switch_nnfw_kernel_env()
{
    local switch=$1  # "ON" or "OFF"
    local mode=$2    # "acl" or "neon" or ""

    # TODO: Handle whether there is nnfw_kernel_env_list.txt or not
    local NNFW_KERNEL_ENV_FILE=$MY_PATH/nnfw_kernel_env_list.txt

    for ENV in $(cat $NNFW_KERNEL_ENV_FILE); do
        if [[ "$switch" == "ON" ]]; then
            export "$ENV=$mode"
        else
            unset "$ENV"
        fi
    done
}
