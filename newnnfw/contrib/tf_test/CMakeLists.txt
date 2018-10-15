nnfw_find_package(Tensorflow QUIET)

if(NOT Tensorflow_FOUND)
  return()
endif(NOT Tensorflow_FOUND)

list(APPEND SOURCES tf_test.cpp)

add_executable(tf_test ${SOURCES})
target_link_libraries(tf_test PRIVATE nnfw_support_tflite)
target_link_libraries(tf_test PRIVATE tensorflow-lite)
target_link_libraries(tf_test PRIVATE tensorflow-core)
