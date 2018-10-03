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

import os
import os.path

mypath = os.path.abspath(os.path.dirname(__file__))


def switch_nnfw_kernel_env(mode):
    # mode  : "acl" or "neon" or ""

    # TODO: Handle whether there is nnfw_kernel_env_list.txt or not
    # FIXME: Now nnfw_kernel_env_list.txt is parent dir of current dir
    filename = "nnfw_kernel_env_list.txt"
    envfilename = mypath + "/../{filename}".format(filename=filename)

    with open(envfilename) as envfile:
        for env in envfile:
            env = env[:-1]  # env has new line at the end
            os.environ[env] = mode


if __name__ == "__main__":
    # for test
    switch_nnfw_kernel_env("acl")
    switch_nnfw_kernel_env("neon")
    switch_nnfw_kernel_env("")
