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
import shutil
import argparse
import common
import subprocess
import sys

mypath = os.path.abspath(os.path.dirname(__file__))


def get_parsed_options():
    parser = argparse.ArgumentParser(prog='test_driver.py', usage='%(prog)s [options]')

    # artifactpath
    parser.add_argument(
        "--artifactpath",
        action="store",
        type=str,
        dest="artifactpath",
        default=".",
        help="(should be passed) path that has tests/ and Product/")

    # test
    parser.add_argument(
        "--unittest",
        action="store_true",
        dest="unittest_on",
        default=False,
        help="(default=on) run unit test")
    parser.add_argument(
        "--unittestall",
        action="store_true",
        dest="unittestall_on",
        default=False,
        help="((default=off) run all unit test without skip, overrite --unittest option")
    parser.add_argument(
        "--verification",
        action="store_true",
        dest="verification_on",
        default=False,
        help="(default=on) run verification")
    parser.add_argument(
        "--frameworktest",
        action="store_true",
        dest="frameworktest_on",
        default=False,
        help="(default=off)run framework test")

    # benchmark
    parser.add_argument(
        "--benchmark",
        action="store_true",
        dest="benchmark_on",
        default=False,
        help="(default=off) run benchmark")
    parser.add_argument(
        "--benchmark_acl",
        action="store_true",
        dest="benchmarkacl_on",
        default=False,
        help="(default=off) run benchmark-acl")
    parser.add_argument(
        "--benchmark_op",
        action="store_true",
        dest="benchmarkop_on",
        default=False,
        help="(default=off) run benchmark per operation")

    # profile
    parser.add_argument(
        "--profile",
        action="store_true",
        dest="profile_on",
        default=False,
        help="(default=off) run profiling")

    # driverbin
    parser.add_argument(
        "--framework_driverbin",
        action="store",
        type=str,
        dest="framework_driverbin",
        help=
        "(default=../../Product/out/bin/tflite_run) runner for runnning framework tests")
    parser.add_argument(
        "--verification_driverbin",
        action="store",
        type=str,
        dest="verification_driverbin",
        help=
        "(default=../../Product/out/bin/nnapi_test) runner for runnning verification tests"
    )
    parser.add_argument(
        "--benchmark_driverbin",
        action="store",
        type=str,
        dest="benchmark_driverbin",
        help=
        "(default=../../Product/out/bin/tflite_benchmark) runner for runnning benchmark")

    # etc.
    parser.add_argument(
        "--runtestsh",
        action="store",
        type=str,
        dest="runtestsh",
        help=
        "(default=ARTIFACT_PATH/tests/framework/run_test.sh) run_test.sh with path where it is for framework test and verification"
    )
    parser.add_argument(
        "--unittestdir",
        action="store",
        type=str,
        dest="unittestdir",
        help=
        "(default=ARTIFACT_PATH/Product/out/unittest) directory that has unittest binaries for unit test"
    )
    parser.add_argument(
        "--ldlibrarypath",
        action="store",
        type=str,
        dest="ldlibrarypath",
        help=
        "(default=ARTIFACT_PATH/Product/out/lib) path that you want to include libraries")
    parser.add_argument(
        "--frameworktest_list_file",
        action="store",
        type=str,
        dest="frameworktest_list_file",
        help=
        "(default=ARTIFACT_PATH/tools/test_driver/pureacl_frameworktest_list.txt) filepath of model list for test"
    )
    parser.add_argument(
        "--reportdir",
        action="store",
        type=str,
        dest="reportdir",
        help="(default=ARTIFACT_PATH/report) directory to save report")

    # env
    parser.add_argument(
        "--usennapi",
        action="store_true",
        dest="usennapi_on",
        default=True,
        help="(default=on)  declare USE_NNAPI=1")
    parser.add_argument(
        "--nousennapi",
        action="store_false",
        dest="usennapi_on",
        help="(default=off) declare nothing about USE_NNAPI")
    parser.add_argument(
        "--acl_envon",
        action="store_true",
        dest="aclenv_on",
        default=False,
        help="(default=off) declare envs for ACL")

    options = parser.parse_args()
    return options


def run_unittest(options):
    cmd = "{artifactpath}/tools/test_driver/run_unittest.sh \
            --reportdir={reportdir} \
            --unittestdir={unittestdir}".format(
        artifactpath=options.artifactpath,
        reportdir=options.reportdir,
        unittestdir=options.unittestdir)
    if options.unittestall_on:
        cmd += " --runall"
    os.system(cmd)


def run_frameworktest(options):
    if type(options.framework_driverbin) is not str:
        options.framework_driverbin = options.artifactpath + "/Product/out/bin/tflite_run"
        if (os.path.exists(options.framework_driverbin) == False):
            print("Cannot find {driverbin}".format(driverbin=options.framework_driverbin))
            sys.exit(1)

    cmd = "{artifactpath}/tools/test_driver/run_frameworktest.sh \
            --runtestsh={runtestsh} \
            --driverbin={driverbin} \
            --reportdir={reportdir} \
            --tapname=framework_test.tap \
            --logname=framework_test.log \
            --testname='Frameworktest'".format(
        runtestsh=options.runtestsh,
        driverbin=options.framework_driverbin,
        reportdir=options.reportdir,
        artifactpath=options.artifactpath)
    os.system(cmd)


def run_verification(options):
    if type(options.verification_driverbin) is not str:
        options.verification_driverbin = options.artifactpath + "/Product/out/bin/nnapi_test"
        if (os.path.exists(options.verification_driverbin) == False):
            print("Cannot find {driverbin}".format(
                driverbin=options.verification_driverbin))
            sys.exit(1)

    cmd = "{artifactpath}/tools/test_driver/run_frameworktest.sh \
            --runtestsh={runtestsh} \
            --driverbin={driverbin} \
            --reportdir={reportdir} \
            --tapname=verification_test.tap \
            --logname=verification_test.log \
            --testname='Verification'".format(
        runtestsh=options.runtestsh,
        driverbin=options.verification_driverbin,
        reportdir=options.reportdir,
        artifactpath=options.artifactpath)
    os.system(cmd)


def run_benchmark(options):
    if type(options.benchmark_driverbin) is not str:
        options.benchmark_driverbin = options.artifactpath + "/Product/out/bin/tflite_benchmark"
        if (os.path.exists(options.benchmark_driverbin) == False):
            print("Cannot find {driverbin}".format(driverbin=options.benchmark_driverbin))
            sys.exit(1)

    cmd = "{artifactpath}/tools/test_driver/run_benchmark.sh \
            --runtestsh={runtestsh} \
            --driverbin={driverbin} \
            --reportdir={reportdir}/benchmark".format(
        runtestsh=options.runtestsh,
        driverbin=options.benchmark_driverbin,
        reportdir=options.reportdir,
        artifactpath=options.artifactpath)
    os.system(cmd)


def run_benchmarkop(options):
    if type(options.benchmark_driverbin) is not str:
        options.benchmark_driverbin = options.artifactpath + "/Product/out/bin/tflite_benchmark"
        if (os.path.exists(options.benchmark_driverbin) == False):
            print("Cannot find {driverbin}".format(driverbin=options.benchmark_driverbin))
            sys.exit(1)

    cmd = "{artifactpath}/tools/test_driver/run_benchmark_op.sh \
            --runtestsh={runtestsh} \
            --driverbin={driverbin} \
            --reportdir={reportdir}/benchmark_op \
            --modelfilepath={artifactpath}/tests/framework \
            --frameworktest_list_file={frameworktest_list_file}".format(
        runtestsh=options.runtestsh,
        driverbin=options.benchmark_driverbin,
        artifactpath=options.artifactpath,
        reportdir=options.reportdir,
        frameworktest_list_file=options.frameworktest_list_file)
    os.system(cmd)


def run_benchmarkacl(options):
    cmd = "{artifactpath}/tools/test_driver/run_benchmark_acl.sh \
            --reportdir={reportdir}/benchmark \
            --bindir={artifactpath}/Product/out/bin".format(
        reportdir=options.reportdir, artifactpath=options.artifactpath)
    os.system(cmd)


def make_json_for_benchmark_result(options):
    cmd = "source {artifactpath}/tools/test_driver/print_to_json.sh && ".format(
        artifactpath=options.artifactpath)
    if options.benchmarkop_on:
        cmd += "print_to_json {artifactpath}/report/benchmark_op \
                {reportdir} \"benchmark_op_result.json\"".format(
            reportdir=options.reportdir, artifactpath=options.artifactpath)
    else:
        cmd += "print_to_json {artifactpath}/report/benchmark \
                {reportdir} \"benchmark_result.json\"".format(
            reportdir=options.reportdir, artifactpath=options.artifactpath)
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", cmd])
    sp.communicate()


def run_profile(options):
    # FIXME: These driver and tflite test are set temporarily. Fix these to support flexibility
    driver_bin = options.artifactpath + "/Product/out/bin/tflite_run"
    tflite_test = options.artifactpath + "/tests/framework/cache/inceptionv3/inception_module/inception_test.tflite"

    # TODO: Enable operf to set directory where sample data puts on
    shutil.rmtree("oprofile_data", ignore_errors=True)

    print("")
    print("============================================")
    cmd = "operf -g {driver_bin} {tflite_test}".format(
        driver_bin=driver_bin, tflite_test=tflite_test)
    os.system(cmd)
    print("============================================")
    print("")


def main():
    options = get_parsed_options()

    alltest_on = True
    if True in [
            options.unittest_on, options.frameworktest_on, options.verification_on,
            options.benchmark_on, options.benchmarkacl_on, options.benchmarkop_on,
            options.profile_on
    ]:
        alltest_on = False

    # artifactpath
    if os.path.isdir(options.artifactpath) and os.path.isdir(
            options.artifactpath + "/tests") and os.path.isdir(options.artifactpath +
                                                               "/Product"):
        options.artifactpath = os.path.abspath(options.artifactpath)
    else:
        print("Pass on with proper arifactpath")
        sys.exit(1)

    # run_test.sh
    if type(options.runtestsh) is not str or options.runtestsh == "":
        options.runtestsh = options.artifactpath + "/tests/framework/run_test.sh"

    if (os.path.exists(options.runtestsh) == False):
        print("Cannot find {runtestsh}".format(runtestsh=options.runtestsh))
        sys.exit(1)

    # unittest dir
    if type(options.unittestdir) is not str or options.unittestdir == "":
        options.unittestdir = options.artifactpath + "/Product/out/unittest"

    # LD_LIBRARY_PATH
    if type(options.ldlibrarypath) is not str or options.ldlibrarypath == "":
        options.ldlibrarypath = options.artifactpath + "/Product/out/lib"

    # report dir
    if type(options.reportdir) is not str or options.reportdir == "":
        options.reportdir = options.artifactpath + "/report"

    # set LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = options.ldlibrarypath

    # set USE_NNAPI
    if options.usennapi_on == True:
        os.environ["USE_NNAPI"] = "1"

    # set acl
    if options.aclenv_on:
        common.switch_nnfw_kernel_env("acl")

    # unittest
    if alltest_on or options.unittest_on:
        run_unittest(options)

    # frameworktest
    if options.frameworktest_on:
        run_frameworktest(options)

    # verification
    if alltest_on or options.verification_on:
        run_verification(options)

    # benchmark
    if options.benchmark_on:
        run_benchmark(options)

    # benchmark_acl
    if options.benchmarkacl_on:
        run_benchmarkacl(options)

    # benchmark_op
    if options.benchmarkop_on:
        run_benchmarkop(options)

    # make json file for benchmark result on ci
    if options.benchmark_on or options.benchmarkacl_on or options.benchmarkop_on:
        make_json_for_benchmark_result(options)

    # run profile
    if options.profile_on:
        run_profile(options)


if __name__ == "__main__":
    main()
