set(TENSORFLOW_BASE ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
set(TENSORFLOW_LITE_BASE ${TENSORFLOW_BASE}/tensorflow/contrib/lite)

# Required source & package
nnfw_find_package(Eigen REQUIRED)
nnfw_find_package(FarmhashSource REQUIRED)
nnfw_find_package(FlatBuffersSource REQUIRED)
nnfw_find_package(GEMMLowpSource REQUIRED)
nnfw_find_package(TensorFlowSource REQUIRED)

#
# Tensorflow Lite library
#
file(GLOB TFLITE_CORE_SRCS "${TENSORFLOW_LITE_BASE}/*.c" "${TENSORFLOW_LITE_BASE}/*.cc")
file(GLOB TFLITE_CORE_TESTS "${TENSORFLOW_LITE_BASE}/*test*.cc")
list(REMOVE_ITEM TFLITE_CORE_SRCS ${TFLITE_CORE_TESTS})

file(GLOB_RECURSE TFLITE_KERNEL_SRCS "${TENSORFLOW_LITE_BASE}/kernels/*.cc")
file(GLOB_RECURSE TFLITE_KERNEL_TESTS "${TENSORFLOW_LITE_BASE}/kernels/*test*.cc")
list(REMOVE_ITEM TFLITE_KERNEL_SRCS ${TFLITE_KERNEL_TESTS})

# We will use our own BuiltinOpResolver
list(REMOVE_ITEM TFLITE_KERNEL_SRCS "${TENSORFLOW_LITE_BASE}/kernels/register.cc")
list(APPEND TFLITE_SRCS ${TFLITE_CORE_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_KERNEL_SRCS})

list(APPEND TFLITE_SRCS "${TFLITE_DEPEND_DIR}/farmhash/src/farmhash.cc")

# Profiling
if(BUILD_TFLITE_BENCHMARK_MODEL)
  file(GLOB TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/*.cc")
  file(GLOB TFLITE_PROFILING_TESTS "${TENSORFLOW_LITE_BASE}/profiling/*test*.cc")
  list(REMOVE_ITEM TFLITE_PROFILING_SRCS ${TFLITE_PROFILING_TESTS})
  list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_BASE}/tensorflow/core/util/stats_calculator.cc")
  list(APPEND TFLITE_SRCS ${TFLITE_PROFILING_SRCS})
endif()

list(APPEND TFLITE_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow")
list(APPEND TFLITE_INCLUDES "${TFLITE_DEPEND_DIR}/gemmlowp")
list(APPEND TFLITE_INCLUDES "${TFLITE_DEPEND_DIR}/farmhash/src")
list(APPEND TFLITE_INCLUDES "${TFLITE_DEPEND_DIR}/flatbuffers/include")

if(BUILD_IS_NATIVE AND NOT HOST_ARCH_BASE STREQUAL "arm")
  # Flatbuffer build
  nnfw_find_package(FlatBuffers)
  # Required external sourcefor x86-64 build
  nnfw_find_package(NEON2SSESource REQUIRED)
  list(APPEND TFLITE_INCLUDES "${TFLITE_DEPEND_DIR}/neon_2_sse")
endif()

# This kernels are not used on nnfw
## spectrogram
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/audio_spectrogram.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/audio_spectrogram_test.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/internal/spectrogram.cc")
## mfcc
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/internal/mfcc.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/internal/mfcc_dct.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/internal/mfcc_mel_filterbank.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/mfcc.cc")
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/kernels/mfcc_test.cc")

if("${TARGET_OS}" STREQUAL "android")
  if(NOT DEFINED NDK_DIR)
    file(GLOB NDK_DIRS "${CMAKE_SOURCE_DIR}/tools/cross/ndk/*")
    list(LENGTH NDK_DIRS NDK_DIRS_COUNT)
    if (NDK_DIRS_COUNT EQUAL 1)
      set(NDK_DIR "${NDK_DIRS}")
    endif(NDK_DIRS_COUNT EQUAL 1)
  endif(NOT DEFINED NDK_DIR)

  if(NOT DEFINED NDK_DIR)
    message(FATAL_ERROR "NDK_DIR should be specified via environment variable")
  endif()
  message(STATUS "Found NDK: ${NDK_DIR}")
  list(APPEND TFLITE_INCLUDES "${NDK_DIR}")
endif()

add_library(tensorflow-lite ${TFLITE_SRCS})
target_include_directories(tensorflow-lite PUBLIC ${TFLITE_INCLUDES})
target_compile_definitions(tensorflow-lite PUBLIC "GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK")
if(BUILD_TFLITE_BENCHMARK_MODEL)
  target_compile_definitions(tensorflow-lite PUBLIC "TFLITE_PROFILING_ENABLED")
endif()
target_link_libraries(tensorflow-lite eigen ${LIB_PTHREAD} dl)

if("${TARGET_OS}" STREQUAL "android")
  target_link_libraries(tensorflow-lite log)
  #
  # Tensorflow Lite JNI library
  #
  set(TFLITE_JNI_BASE ${TENSORFLOW_LITE_BASE}/java/src/main/native)
  set(TFLITE_JNI_SRCS ${TFLITE_JNI_BASE}/duration_utils_jni.cc
                      ${TFLITE_JNI_BASE}/exception_jni.cc
                      ${TFLITE_JNI_BASE}/nativeinterpreterwrapper_jni.cc
                      ${TFLITE_JNI_BASE}/tensor_jni.cc
                      ${TFLITE_JNI_BASE}/tensorflow_lite_jni.cc
                      ${TFLITE_JNI_BASE}/builtin_ops_jni.cc
                      )
  set(TFLITE_JNI_INCLUDES ${TENSORFLOW_LITE_BASE}/java/src/native)

  # We need this for running vanilla tflite
  # TODO remove this when nnfw is used
  set(TFLITE_SRCS_V ${TENSORFLOW_LITE_BASE}/kernels/register.cc)

  # TODO use tensorflow-lite static library instead of compiling all the sources again
  add_library(tensorflowlite_jni SHARED ${TFLITE_JNI_SRCS} ${TFLITE_SRCS} ${TFLITE_SRCS_V})
  target_include_directories(tensorflowlite_jni PUBLIC ${TFLITE_JNI_INCLUDES} ${TFLITE_INCLUDES})
  target_link_libraries(tensorflowlite_jni eigen ${LIB_PTHREAD} dl)
  target_link_libraries(tensorflowlite_jni log)
  install(TARGETS tensorflowlite_jni DESTINATION lib)
endif()
