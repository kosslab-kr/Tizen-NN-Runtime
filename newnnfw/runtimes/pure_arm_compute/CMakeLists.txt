if(NOT BUILD_PURE_ARM_COMPUTE)
  return()
endif(NOT BUILD_PURE_ARM_COMPUTE)

nnfw_find_package(ARMCompute REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cc")

add_library(nnapi_pure_arm_compute SHARED ${SOURCES})
# To ignore compiler warnings from ARM Compute Library
target_compile_options(nnapi_pure_arm_compute PRIVATE -Wno-ignored-attributes)
target_include_directories(nnapi_pure_arm_compute PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(nnapi_pure_arm_compute PUBLIC src)
if(BUILD_TFLITE_BENCHMARK_MODEL)
  target_compile_definitions(nnapi_pure_arm_compute PUBLIC "TFLITE_PROFILING_ENABLED")
endif()
target_link_libraries(nnapi_pure_arm_compute arm_compute)
target_link_libraries(nnapi_pure_arm_compute arm_compute_ex)
target_link_libraries(nnapi_pure_arm_compute static_nnfw_util)

if("${TARGET_OS}" STREQUAL "android")
  target_link_libraries(nnapi_pure_arm_compute log)
endif()

set_target_properties(nnapi_pure_arm_compute PROPERTIES OUTPUT_NAME neuralnetworks)
install(TARGETS nnapi_pure_arm_compute DESTINATION lib/pureacl RENAME neuralnetworks)
install(TARGETS nnapi_pure_arm_compute DESTINATION lib RENAME neuralnetworks)

# To prevent undefined references
add_executable(pure_arm_compute_symbolcheck symbolcheck.cpp)
target_link_libraries(pure_arm_compute_symbolcheck nnapi_pure_arm_compute)
