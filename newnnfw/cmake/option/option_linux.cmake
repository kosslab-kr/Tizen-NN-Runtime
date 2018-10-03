#
# linux common compile options
#

# flags for build type: debug, release
set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE   "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# test-coverage build flag
if("${COVERAGE_BUILD}" STREQUAL "1")
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  set(FLAGS_COMMON "${FLAGS_COMMON} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

#
# linux common variable and settings
#

# lib pthread as a variable (pthread must be disabled on android)
set(LIB_PTHREAD pthread)

# nnfw common path
set(NNFW_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(NNFW_EXTERNALS_DIR ${CMAKE_SOURCE_DIR}/externals)

# External sources to build tflite
# If already downloaded files are in tensorflow/tensorflow/contrib/lite/downloads,
# set TFLITE_DEPEND_DIR to ${NNFW_EXTERNALS_DIR}/tensorflow/tensorflow/contrib/lite/downloads
set(TFLITE_DEPEND_DIR ${NNFW_EXTERNALS_DIR})
