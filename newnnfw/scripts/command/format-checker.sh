#!/bin/bash

function pushd () {
    command pushd "$@" > /dev/null
}

function popd () {
    command popd "$@" > /dev/null
}

function check_cpp_tool() {
    which clang-format-3.9
    if [[ $? -ne 0 ]]; then
        echo "Error: clang-format-3.9 is not available."
        echo "       Please install clang-format-3.9."
        exit 1
    fi
}

function check_python_tool() {
    which yapf
    if [[ $? -ne 0 ]]; then
        echo "Error: yapf is not available."
        echo "       Please install yapf."
        exit 1
    fi
}

function check_cpp_files() {
    DIRECTORIES_TO_BE_TESTED=$1
    DIRECTORIES_NOT_TO_BE_TESTED=$2

    # Check c++ files
    for TEST_DIR in ${DIRECTORIES_TO_BE_TESTED[@]}; do
        pushd $TEST_DIR
            CPP_FILES_TO_CHECK=$(git ls-files '*.h' '*.cpp' '*.cc')
            ARR=($CPP_FILES_TO_CHECK)
            for s in ${DIRECTORIES_NOT_TO_BE_TESTED[@]}; do
                if [[ $s = $TEST_DIR* ]]; then
                    skip=${s#$TEST_DIR/}/
                    ARR=(${ARR[*]//$skip*})
                fi
            done
            CPP_FILES_TO_CHECK=${ARR[*]}
            if [[ ${#CPP_FILES_TO_CHECK} -ne 0 ]]; then
                clang-format-3.9 -i $CPP_FILES_TO_CHECK
            fi
        popd
    done
}

function check_python_files() {
    DIRECTORIES_TO_BE_TESTED=$1
    DIRECTORIES_NOT_TO_BE_TESTED=$2

    # Check python files
    for TEST_DIR in ${DIRECTORIES_TO_BE_TESTED[@]}; do
        pushd $TEST_DIR
            PYTHON_FILES_TO_CHECK=$(git ls-files '*.py')
            ARR=($PYTHON_FILES_TO_CHECK)
            for s in ${DIRECTORIES_NOT_TO_BE_TESTED[@]}; do
                if [[ $s = $TEST_DIR* ]]; then
                    skip=${s#$TEST_DIR/}/
                    ARR=(${ARR[*]//$skip*})
                fi
            done
            PYTHON_FILES_TO_CHECK=${ARR[*]}
            if [[ ${#PYTHON_FILES_TO_CHECK} -ne 0 ]]; then
                yapf -i --style='{based_on_style: pep8, column_limit: 90}' $PYTHON_FILES_TO_CHECK
            fi
        popd
    done
}

echo "Make sure commit all changes before running this checker."

__Check_CPP=${CHECK_CPP:-"1"}
__Check_PYTHON=${CHECK_PYTHON:-"1"}

DIRECTORIES_TO_BE_TESTED=()
DIRECTORIES_NOT_TO_BE_TESTED=()

for DIR_TO_BE_TESTED in $(find -name '.FORMATCHECKED' -exec dirname {} \;); do
    DIRECTORIES_TO_BE_TESTED+=("$DIR_TO_BE_TESTED")
done

for DIR_NOT_TO_BE_TESTED in $(find -name '.FORMATDENY' -exec dirname {} \;); do
    DIRECTORIES_NOT_TO_BE_TESTED+=("$DIR_NOT_TO_BE_TESTED")
done

if [[ ${#DIRECTORIES_TO_BE_TESTED[@]} -eq 0 ]]; then
    echo "No directories to be checked"
    exit 0
fi

if [[ $__Check_CPP -ne 0 ]]; then
  check_cpp_tool
  check_cpp_files $DIRECTORIES_TO_BE_TESTED $DIRECTORIES_NOT_TO_BE_TESTED
fi

if [[ $__Check_PYTHON -ne 0 ]]; then
  check_python_tool
  check_python_files $DIRECTORIES_TO_BE_TESTED $DIRECTORIES_NOT_TO_BE_TESTED
fi

git diff --ignore-submodules > format.patch
PATCHFILE_SIZE=$(stat -c%s format.patch)
if [[ $PATCHFILE_SIZE -ne 0 ]]; then
    echo "[FAILED] Format checker failed and update code to follow convention."
    echo "         You can find changes in format.patch"
    exit 1
else
    echo "[PASSED] Format checker succeed."
    exit 0
fi

echo "Error: Something went wrong."
exit 1
