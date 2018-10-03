set(NNRUNTIME_TEST_INC_COMMON ${NNFW_INCLUDE_DIR}
                              ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (BUILD_ANDROID_NN_RUNTIME_TEST)
  add_subdirectory(neural_networks_test)
endif()
