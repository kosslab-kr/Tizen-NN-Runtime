# Executable `runtime_run` (Dummy runner executable using NN API)
set(RUNTIME_ANDROID_NN_TEST runtime_run_android_nn_test)
set(RUNTIME_ANDROID_NN_TEST_SRC TestMain.cpp
                                TestTrivialModel.cpp
                                TestValidation.cpp)

# g++-6.3 can build TestGenerated.cpp.
if (CMAKE_COMPILER_IS_GNUCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.3)
  set(RUNTIME_ANDROID_NN_TEST_SRC ${RUNTIME_ANDROID_NN_TEST_SRC} TestGenerated.cpp)
endif()

# Both gbs and cross for tizen support building generated test
if (OBS_BUILD)
  set(RUNTIME_ANDROID_NN_TEST_SRC ${RUNTIME_ANDROID_NN_TEST_SRC} TestGenerated.cpp)
endif()

add_executable(${RUNTIME_ANDROID_NN_TEST} ${RUNTIME_ANDROID_NN_TEST_SRC})

nnfw_find_package(GTest)

target_include_directories(${RUNTIME_ANDROID_NN_TEST} PRIVATE
                              ${NNRUNTIME_TEST_INC_COMMON}
                              ${CMAKE_CURRENT_SOURCE_DIR}
                              ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (BUILD_NEURUN)
  target_link_libraries(${RUNTIME_ANDROID_NN_TEST} ${LIB_NEURUN})
  set(SKIPLIST_FILE_NAME ${RUNTIME_ANDROID_NN_TEST}.skip.${TARGET_PLATFORM}.neurun)
elseif (BUILD_PURE_ARM_COMPUTE)
  target_link_libraries(${RUNTIME_ANDROID_NN_TEST} ${LIB_PUREACL_RUNTIME})
  set(SKIPLIST_FILE_NAME ${RUNTIME_ANDROID_NN_TEST}.skip.${TARGET_PLATFORM})
endif()
target_link_libraries(${RUNTIME_ANDROID_NN_TEST} gtest)
target_link_libraries(${RUNTIME_ANDROID_NN_TEST} ${LIB_PTHREAD})

install(TARGETS runtime_run_android_nn_test DESTINATION unittest)

install(FILES ${SKIPLIST_FILE_NAME}
        DESTINATION unittest
        RENAME ${RUNTIME_ANDROID_NN_TEST}.skip
        OPTIONAL)
