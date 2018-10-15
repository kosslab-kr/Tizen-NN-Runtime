file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(nnfw_support_tflite STATIC ${SOURCES})
set_target_properties(nnfw_support_tflite PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_support_tflite PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nnfw_support_tflite tensorflow-lite ${LIB_PTHREAD} dl)
target_link_libraries(nnfw_support_tflite static_nnfw_util)

add_executable(nnfw_support_tflite_test_TensorView src/TensorView.test.cpp)
target_link_libraries(nnfw_support_tflite_test_TensorView nnfw_support_tflite)
