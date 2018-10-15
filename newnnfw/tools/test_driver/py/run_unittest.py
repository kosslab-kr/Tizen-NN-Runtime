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
import subprocess


def get_parsed_options():
    parser = argparse.ArgumentParser(prog='run_unittest.py', usage='%(prog)s [options]')

    parser.add_argument(
        "--reportdir",
        action="store",
        type=str,
        dest="reportdir",
        default="report",
        help="(default=report) directory that each test result will be stored")

    parser.add_argument(
        "--unittestdir",
        action="store",
        type=str,
        dest="unittestdir",
        required=True,
        help="directory that unittests are included")

    parser.add_argument(
        "--ldlibrarypath",
        action="store",
        type=str,
        dest="ldlibrarypath",
        help=
        "(usually : ARTIFACT_PATH/Product/out/lib) path that you want to include libraries"
    )

    parser.add_argument(
        "--runall",
        action="store_true",
        dest="runall",
        default=False,
        help="run all unittest and ignore skiplist")

    options = parser.parse_args()
    return options


def get_gtest_option(report_dir, test_bin, unittest_dir=None):
    # Set path to save test result
    output_option = "--gtest_output=xml:{report_dir}/{test_bin}.xml".format(
        report_dir=report_dir, test_bin=test_bin)

    # Set filter to run only one unit test, for runall unittest
    if '.' in test_bin:
        return output_option + " " + "--gtest_filter={test_list_item}".format(
            test_list_item=test_bin)

    # Set filter not to run *.skip unit tests
    filter_option = ""
    skiplist_path = "{unittest_dir}/{test_bin}.skip".format(
        unittest_dir=unittest_dir, test_bin=test_bin)
    if os.path.exists(skiplist_path):
        filter_option = "--gtest_filter=-"
        skiplist_file = open(skiplist_path, "r")
        filter_option = filter_option + ':'.join(line[:-1] for line in skiplist_file
                                                 if line[0] != '#')
        skiplist_file.close()

    return output_option + " " + filter_option


def get_test_list_items(unittest_dir, test_bin):
    cmd_output = subprocess.check_output(
        "{unittestdir}/{testbin} --gtest_list_tests".format(
            unittestdir=unittest_dir, testbin=test_bin),
        shell=True)
    all_test_list = str(cmd_output).replace('\\n', ' ').split()
    all_test_list[0] = all_test_list[0][2:]

    category = ""
    item = ""
    test_list_items = []
    for verbose_line in all_test_list:
        if verbose_line[-1] == '.':
            category = verbose_line
        else:
            item = "{category}{verbose_line}".format(
                category=category, verbose_line=verbose_line)
            test_list_items.append(item)

    return test_list_items


# Just call this function when running unit test in test_driver.py
def run_unittest(unittest_dir, report_dir, ldlibrary_path, runall):
    if unittest_dir == "" or unittest_dir == None:
        print("Fail : unittestdir is not given")
        print("(Usually unit test directory is Product/out/unittest)")
        sys.exit(1)

    if report_dir == "" or report_dir == None:
        print("Info : 'report' folder of current path will be used as report directory")
        report_dir = "report"

    if type(ldlibrary_path) is str and ldlibrary_path != "":
        os.environ["LD_LIBRARY_PATH"] = ldlibrary_path

    print("")
    print("============================================")
    print("Unittest start")
    print("============================================")

    # Run all unit tests in unittest_dir
    unittest_result = 0
    all_test_bin = (t for t in os.listdir(unittest_dir)
                    if len(t) < 5 or t[-5:] != ".skip")

    for idx, test_bin in enumerate(all_test_bin):
        num_unittest = idx + 1
        print("============================================")
        print("Starting set {num_unittest}: {test_bin}...".format(
            num_unittest=num_unittest, test_bin=test_bin))
        print("============================================")

        ret = 0

        # Run all unit tests ignoring skip list
        if runall:
            test_list_items = get_test_list_items(unittest_dir, test_bin)
            for test_item in test_list_items:
                cmd = "{unittest_dir}/{test_bin} {gtest_option}".format(
                    unittest_dir=unittest_dir,
                    test_bin=test_bin,
                    gtest_option=get_gtest_option(report_dir, test_item))
                os.system(cmd)
        # Run all unit tests except skip list
        else:
            cmd = "{unittest_dir}/{test_bin} {gtest_option}".format(
                unittest_dir=unittest_dir,
                test_bin=test_bin,
                gtest_option=get_gtest_option(report_dir, test_bin, unittest_dir))
            ret = os.system(cmd)

        if ret != 0:
            unittest_result = ret
            print("{test_bin} failed... return code: {unittest_result}".format(
                test_bin=test_bin, unittest_result=unittest_result))

        print("============================================")
        print("Finishing set {num_unittest}: {test_bin}...".format(
            num_unittest=num_unittest, test_bin=test_bin))
        print("============================================")

    if unittest_result != 0:
        print("============================================")
        print("Failed unit test... exit code: {unittest_result}".format(
            unittest_result=unittest_result))
        print("============================================")
        sys.exit(1)

    print("============================================")
    print("Completed total {num_unittest} set of unittest".format(
        num_unittest=num_unittest))
    print("Unittest end")
    print("============================================")
    sys.exit(0)


if __name__ == "__main__":
    options = get_parsed_options()
    sys.exit(
        run_unittest(options.unittestdir, options.reportdir, options.ldlibrarypath,
                     options.runall))
