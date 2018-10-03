list(APPEND TFLITE_RUN_SRCS "src/tflite_run.cc")
list(APPEND TFLITE_RUN_SRCS "src/bin_image.cc")
list(APPEND TFLITE_RUN_SRCS "src/args.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_dumper.cc")
list(APPEND TFLITE_RUN_SRCS "src/tensor_loader.cc")

add_executable(tflite_run ${TFLITE_RUN_SRCS})
target_include_directories(tflite_run PRIVATE src)
target_link_libraries(tflite_run tensorflow-lite ${LIB_PTHREAD} dl nnfw_support_tflite)
target_link_libraries(tflite_run boost_program_options boost_system boost_filesystem)

install(TARGETS tflite_run DESTINATION bin)

# TEST BUILD
nnfw_find_package(GTest)

if(NOT GTest_FOUND)
  return()
endif(NOT GTest_FOUND)

## Add test cpp file
add_executable(tflite_test src/tflite_test.cc)
## Link test executable against gtest & gtest_main
target_link_libraries(tflite_test gtest gtest_main ${LIB_PTHREAD})
## install test binary for packaging
install(TARGETS tflite_test DESTINATION unittest)
