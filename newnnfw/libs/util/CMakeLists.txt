# Library `nnfw_util`
set(NNFW_UTILITY_SRCS src/environment.cpp)
list(APPEND NNFW_UTILITY_SRCS src/tensor/Shape.cpp)
list(APPEND NNFW_UTILITY_SRCS src/tensor/NonIncreasingStride.cpp)
list(APPEND NNFW_UTILITY_SRCS src/tensor/IndexFormatter.cpp)
list(APPEND NNFW_UTILITY_SRCS src/tensor/Comparator.cpp)
if(BUILD_TFLITE_BENCHMARK_MODEL)
  list(APPEND NNFW_UTILITY_SRCS src/profiling/time.cc)
endif()

add_library(nnfw_util SHARED ${NNFW_UTILITY_SRCS})
target_include_directories(nnfw_util PUBLIC ${NNFW_INCLUDE_DIR})

add_library(static_nnfw_util STATIC ${NNFW_UTILITY_SRCS})
target_include_directories(static_nnfw_util PUBLIC ${NNFW_INCLUDE_DIR})
set_target_properties(static_nnfw_util PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS nnfw_util
        RUNTIME DESTINATION bin COMPONENT libraries
        LIBRARY DESTINATION lib COMPONENT libraries)

add_executable(nnfw_util_tensor_index_iterator "examples/tensor_index_iterator.cpp")
target_link_libraries(nnfw_util_tensor_index_iterator nnfw_util)
