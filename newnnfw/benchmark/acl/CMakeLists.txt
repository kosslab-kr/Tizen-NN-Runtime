nnfw_find_package(ARMCompute REQUIRED)

add_library(arm_compute_benchmark SHARED "Benchmark.cpp")
target_include_directories(arm_compute_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(arm_compute_benchmark arm_compute_graph)
install(TARGETS arm_compute_benchmark DESTINATION lib)

# GoogLeNet benchmark
add_executable(benchmark_googlenet "benchmark_googlenet.cpp")
target_link_libraries(benchmark_googlenet arm_compute_benchmark)

# GoogLeNet benchmark
add_executable(benchmark_inception_v3 "benchmark_inception_v3.cpp")
target_link_libraries(benchmark_inception_v3 arm_compute_benchmark)

# MobileNet benchmark
add_executable(benchmark_mobilenet "benchmark_mobilenet.cpp")
target_link_libraries(benchmark_mobilenet arm_compute_benchmark)

install(TARGETS benchmark_googlenet benchmark_inception_v3 benchmark_mobilenet DESTINATION bin)
