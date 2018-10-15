#!/bin/bash

# This file is based on https://github.sec.samsung.net/STAR/nncc/pull/80

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOST_HOME=$SCRIPT_ROOT/../..

LCOV_PATH=$(command -v lcov)
GENHTML_PATH=$(command -v genhtml)


SRC_PREFIX=${SRC_PREFIX:-${DOCKER_HOME}}

if [[ -z "${LCOV_PATH}" ]]; then
  echo "ERROR: 'lcov' is not found"
  exit 255
fi

if [[ -z "${GENHTML_PATH}" ]]; then
  echo "ERROR: 'genhtml' is not found"
  exit 255
fi

OUTPUT_PATH="$1"

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="$HOST_HOME/coverage"
fi

if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "ERROR: '${OUTPUT_PATH}' already exists"
  exit 255
fi

mkdir -p "${OUTPUT_PATH}"

RAW_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.raw.info"
LIBS_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.libs.info"
INCLUDE_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.include.info"
RUNTIMES_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.runtimes.info"
TOOLS_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.tools.info"
FINAL_COVERAGE_INFO_PATH="${OUTPUT_PATH}/coverage.info"
HTML_PATH="${OUTPUT_PATH}/html"
COVERTURA_PATH="${OUTPUT_PATH}/nnfw_coverage.xml"

"${LCOV_PATH}" -c -d "${HOST_HOME}" -o "${RAW_COVERAGE_INFO_PATH}"
"${LCOV_PATH}" -e "${RAW_COVERAGE_INFO_PATH}" -o "${LIBS_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/libs/*"
"${LCOV_PATH}" -e "${RAW_COVERAGE_INFO_PATH}" -o "${INCLUDE_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/include/*"
"${LCOV_PATH}" -e "${RAW_COVERAGE_INFO_PATH}" -o "${RUNTIMES_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/runtimes/*"
"${LCOV_PATH}" -e "${RAW_COVERAGE_INFO_PATH}" -o "${TOOLS_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/tools/*"
"${LCOV_PATH}" -a "${LIBS_COVERAGE_INFO_PATH}" -a "${INCLUDE_COVERAGE_INFO_PATH}" \
               -a "${RUNTIMES_COVERAGE_INFO_PATH}" -a "${TOOLS_COVERAGE_INFO_PATH}" \
               -o "${FINAL_COVERAGE_INFO_PATH}"
"${LCOV_PATH}" -r "${FINAL_COVERAGE_INFO_PATH}" -o "${FINAL_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/runtimes/tests/*"
"${LCOV_PATH}" -r "${FINAL_COVERAGE_INFO_PATH}" -o "${FINAL_COVERAGE_INFO_PATH}" "${SRC_PREFIX}/runtimes/nn/depend/*"
"${GENHTML_PATH}" "${FINAL_COVERAGE_INFO_PATH}" --output-directory "${HTML_PATH}" ${GENHTML_FLAG:-}

tar -zcf "${OUTPUT_PATH}"/coverage_report.tar.gz "${HTML_PATH}"
$SCRIPT_ROOT/lcov-to-covertura-xml.sh "${FINAL_COVERAGE_INFO_PATH}" -o "${COVERTURA_PATH}"
