#!/usr/bin/env python

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
import sys
import argparse


def get_parsed_options():
    parser = argparse.ArgumentParser(
        prog='run_frameworktest.py', usage='%(prog)s [options]')

    parser.add_argument(
        "--runtestsh",
        action="store",
        type=str,
        dest="fwtest_runtestsh",
        required=True,
        help="(Usually : tests/framework/run_test.sh) run test shell for framework test")

    parser.add_argument(
        "--driverbin",
        action="store",
        type=str,
        dest="fwtest_driverbin",
        required=True,
        help="(Usually in Product/out/bin/) driver bin for framework test")

    parser.add_argument(
        "--tapname",
        action="store",
        type=str,
        dest="fwtest_tapname",
        help="tap name for framework test")

    parser.add_argument(
        "--logname",
        action="store",
        type=str,
        dest="fwtest_logname",
        help="log name for framework test")

    parser.add_argument(
        "--testname",
        action="store",
        type=str,
        dest="fwtest_testname",
        help="test name of framework test")

    parser.add_argument(
        "--frameworktest_list_file",
        action="store",
        type=str,
        dest="frameworktest_list_file",
        help="list of files to run framework test")

    parser.add_argument(
        "--reportdir",
        action="store",
        type=str,
        dest="fwtest_reportdir",
        default="report",
        help="(default=report) directory that each test result will be stored")

    parser.add_argument(
        "--ldlibrarypath",
        action="store",
        type=str,
        dest="ldlibrarypath",
        help=
        "(usually : ARTIFACT_PATH/Product/out/lib) path that you want to include libraries"
    )

    options = parser.parse_args()
    return options


# Check each parameters if they are valid or not
def check_params(fwtest_runtestsh, fwtest_driverbin, fwtest_reportdir, fwtest_tapname,
                 fwtest_logname, fwtest_testname, frameworktest_list_file,
                 ldlibrary_path):
    if fwtest_runtestsh == "" or fwtest_runtestsh == None:
        print("Fail : runtestsh is not given")
        print("(Usually runtestsh for framework test is tests/framework/run_test.sh)")
        sys.exit(1)

    if os.path.isfile(fwtest_runtestsh) == False:
        print("Fail : runtestsh is not valid")
        sys.exit(1)

    if fwtest_driverbin == "" or fwtest_driverbin == None:
        print("Fail : driverbin is not given")
        print("(Usually driverbin for framework test is in Product/out/bin/)")
        sys.exit(1)

    if os.path.isfile(fwtest_driverbin) == False:
        print("Fail : driverbin is not valid")
        sys.exit(1)

    if fwtest_testname == "" or fwtest_testname == None:
        print("Fail : testname is not given")
        sys.exit(1)

    if fwtest_tapname == "" or fwtest_tapname == None:
        print("Fail : tapname is not given")
        sys.exit(1)

    if fwtest_logname == "" or fwtest_logname == None:
        print("Fail : logname is not given")
        sys.exit(1)

    if fwtest_reportdir == "" or fwtest_reportdir == None:
        print("Fail : report directory is not given")
        sys.exit(1)

    if type(ldlibrary_path) is str and ldlibrary_path != "":
        os.environ["LD_LIBRARY_PATH"] = ldlibrary_path


# Just call this function when running framework test in test_driver.py
def run_frameworktest(fwtest_runtestsh, fwtest_driverbin, fwtest_reportdir,
                      fwtest_tapname, fwtest_logname, fwtest_testname,
                      frameworktest_list_file, ldlibrary_path):

    # Handling exceptions for parameters
    check_params(fwtest_runtestsh, fwtest_driverbin, fwtest_reportdir, fwtest_tapname,
                 fwtest_logname, fwtest_testname, frameworktest_list_file, ldlibrary_path)

    os.makedirs(fwtest_reportdir, exist_ok=True)

    print("")
    print("============================================")
    print("{fwtest_testname} with {fwtest_driverbin_name} ...".format(
        fwtest_testname=fwtest_testname,
        fwtest_driverbin_name=fwtest_driverbin[fwtest_driverbin.rfind('/') + 1:]))

    # Run framework test using models in model_list
    model_list = ""
    if frameworktest_list_file != None and frameworktest_list_file != "":
        fwtest_list_file = open(frameworktest_list_file, "r")
        for line in fwtest_list_file:
            model_list += (line[:-1] + " ")
        fwtest_list_file.close()

    # If model_list is empty, all possible models will be found automatically by fwtest_runtestsh
    cmd = "{fwtest_runtestsh} --driverbin={fwtest_driverbin} \
    --reportdir={fwtest_reportdir} \
    --tapname={fwtest_tapname} \
    {model_list} \
    > {fwtest_reportdir}/{fwtest_logname} 2>&1".format(
        fwtest_runtestsh=fwtest_runtestsh,
        fwtest_driverbin=fwtest_driverbin,
        fwtest_reportdir=fwtest_reportdir,
        fwtest_tapname=fwtest_tapname,
        model_list=model_list,
        fwtest_logname=fwtest_logname)
    fwtest_result = os.system(cmd)

    print("")
    tap_file_path = "{fwtest_reportdir}/{fwtest_tapname}".format(
        fwtest_reportdir=fwtest_reportdir, fwtest_tapname=fwtest_tapname)
    tap_file = open(tap_file_path, "r")
    tap_data = tap_file.read()
    print(tap_data)
    tap_file.close()

    if fwtest_result != 0:
        print("")
        print("{fwtest_testname} failed... exit code: {fwtest_result}".format(
            fwtest_testname=fwtest_testname, fwtest_result=fwtest_result))
        print("============================================")
        print("")
        sys.exit(1)

    print("============================================")
    print("")
    sys.exit(0)


if __name__ == "__main__":
    options = get_parsed_options()
    sys.exit(
        run_frameworktest(options.fwtest_runtestsh, options.fwtest_driverbin,
                          options.fwtest_reportdir, options.fwtest_tapname,
                          options.fwtest_logname, options.fwtest_testname,
                          options.frameworktest_list_file, options.ldlibrarypath))
