#
# Configuration flags
#
option(BUILD_ACL "Build ARM Compute Library" OFF)
option(BUILD_PURE_ARM_COMPUTE "Build pure_arm_compute runtime" ON)
option(BUILD_ACL_STATIC_LIB "Build ARM Comput Static Library" OFF)
option(BUILD_BENCHMARK_ACL "Build ARM Compute Library Benchmarks" OFF)
option(BUILD_NEURUN "Build neurun" OFF) #if implementation is done, it would replace nn runtime.
option(BUILD_LABS "Build lab projects" ON)
option(BUILD_ANDROID_NN_RUNTIME_TEST "Build Android NN Runtime Test" ON)
option(BUILD_DETECTION_APP "Build detection example app" OFF)
option(BUILD_NNAPI_QUICKCHECK "Build NN API Quickcheck tools" OFF)
option(BUILD_TFLITE_BENCHMARK_MODEL "Build tflite benchmark model" OFF)

if("${TARGET_ARCH}" STREQUAL "armv7l" AND NOT "${TARGET_OS}" STREQUAL "tizen")
  set(BUILD_PURE_ARM_COMPUTE ON)
endif()

# On x86, disable pureacl/new runtine build which depends on arm compute library
if("${TARGET_ARCH}" STREQUAL "x86_64")
  set(BUILD_PURE_ARM_COMPUTE OFF)
  set(BUILD_NEW_RUNTIME OFF)
endif()
