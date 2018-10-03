cmake_minimum_required(VERSION 3.1)

project(nnfw)

macro(nnfw_include PREFIX)
  include("${CMAKE_SOURCE_DIR}/cmake/modules/${PREFIX}.cmake")
endmacro(nnfw_include)

macro(nnfw_find_package PREFIX)
  find_package(${PREFIX} CONFIG NO_DEFAULT_PATH PATHS ${CMAKE_SOURCE_DIR}/cmake/packages ${ARGN})
endmacro(nnfw_find_package)

set(CMAKE_CXX_STANDARD 11)

# identify platform: HOST_PLATFORM, TARGET_PLATFORM and related
include("cmake/option/identify_platform.cmake")

# platform specific options
include("cmake/option/option_${TARGET_PLATFORM}.cmake")

# apply compilation flags
# note: this should be placed after cmake/option/option_xxx.cmake files
include("cmake/ApplyCompileFlags.cmake")

# Configuration flags
include("cmake/CfgOptionFlags.cmake")
# and besides CfgOptionFlags.cmake that can be given outside
#   OBS_BUILD: build boolean flag that tizen in OBS build
#   COVERAGE_BUILD: build boolean flag that enables converage test
#   ROOTFS_ARM: arm rootfs path for cross building
#   ROOTFS_ARM64: arm 64bit rootfs path for cross building, linux,tizen,android
#   TARGET_ARCH: target architecture string for cross building
#   TARGET_OS: target os string for cross building

# NOTE '${CMAKE_INSTALL_PREFIX}/lib' should be added as a link directory as
#      CI server places pre-built ARM compute libraries on this directory.
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

# Download configuration
option(DOWNLOAD_TENSORFLOW "Download Tensorflow source" ON)
option(DOWNLOAD_EIGEN "Download Eigen source" ON)
option(DOWNLOAD_FARMHASH "Download farmhash source" ON)
option(DOWNLOAD_GEMMLOWP "Download GEMM low precesion library source" ON)
option(DOWNLOAD_NEON2SSE "Download NEON2SSE library source" ON)
option(DOWNLOAD_FLATBUFFERS "Download FlatBuffers source" ON)

# GTest support
option(BUILD_GTEST "Download and build Google Test" ON)
nnfw_find_package(GTest QUIET)

# TODO For now Android build is being enabled incrementally so not all subdirectories are added yet.
#      However we are going to have the same subdirectories with other OS eventually.
if("${TARGET_OS}" STREQUAL "android")

  include_directories(externals/tensorflow)
  include_directories(externals/acl)
  include_directories(externals/acl/include)
  include_directories(externals/flatbuffers/include)
  include_directories(include)

  add_subdirectory(libs)
  add_subdirectory(externals)
  add_subdirectory(tools/nnapi_test)
  add_subdirectory(tools/tflite_benchmark)

  add_subdirectory(runtimes)

else("${TARGET_OS}" STREQUAL "android") # General case (non-android build)

  if(NOT OBS_BUILD)
    add_subdirectory(externals)
    add_subdirectory(tools)
    add_subdirectory(benchmark)
    add_subdirectory(contrib)
  else(NOT OBS_BUILD)
    add_subdirectory(tools/nnapi_test)
  endif(NOT OBS_BUILD)

  add_subdirectory(libs)
  add_subdirectory(runtimes)

endif("${TARGET_OS}" STREQUAL "android")
