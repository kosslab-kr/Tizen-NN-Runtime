file(GLOB_RECURSE SOURCES "*.cc")

add_executable(tflite_benchmark_model ${SOURCES})
target_compile_definitions(tflite_benchmark_model PUBLIC "TFLITE_PROFILING_ENABLED")
target_link_libraries(tflite_benchmark_model tensorflow-lite ${LIB_PTHREAD} dl nnfw_util nnfw_support_tflite)
install(TARGETS tflite_benchmark_model DESTINATION bin)
