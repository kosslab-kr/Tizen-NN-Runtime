list(APPEND SOURCES "src/tflite_benchmark.cc")

add_executable(tflite_benchmark ${SOURCES})
target_link_libraries(tflite_benchmark tensorflow-lite ${LIB_PTHREAD} dl nnfw_util nnfw_support_tflite)
install(TARGETS tflite_benchmark DESTINATION bin)
