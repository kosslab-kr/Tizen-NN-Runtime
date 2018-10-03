# Add cpu

set(NEURUN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Build kernels
set(LIB_NEURUN_KERNEL_CPU neurun_kernel_cpu)
set(LIB_NEURUN_KERNEL_ACL_CL neurun_kernel_acl_cl)
add_subdirectory(src/kernel)

# Build backends
set(LIB_NEURUN_BACKEND_CPU neurun_backend_cpu)
set(LIB_NEURUN_BACKEND_ACL_CL neurun_backend_acl_cl)
add_subdirectory(src/backend)

file(GLOB SOURCES "src/*.cc")
file(GLOB_RECURSE SOURCES_FRONTEND "src/frontend/*.cc")
file(GLOB SOURCES_BACKEND "src/backend/*.cc")
file(GLOB_RECURSE SOURCES_INTERNAL "src/internal/*.cc")
file(GLOB_RECURSE SOURCES_GRAPH "src/graph/*.cc")
file(GLOB_RECURSE SOURCES_LINEAR "src/linear/*.cc")
file(GLOB_RECURSE SOURCES_CODEGEN "src/codegen/*.cc")
file(GLOB_RECURSE SOURCES_VERIFIER "src/verifier/*.cc")

set(SOURCES ${SOURCES} ${SOURCES_FRONTEND} ${SOURCES_BACKEND} ${SOURCES_INTERNAL} ${SOURCES_GRAPH} ${SOURCES_LINEAR} ${SOURCES_CODEGEN} ${SOURCES_VERIFIER})

# NOTE For now ARMCompute is necessary
# TODO Remove required package below(should be optional)
nnfw_find_package(ARMCompute REQUIRED)

add_library(${LIB_NEURUN} SHARED ${SOURCES})
target_include_directories(${LIB_NEURUN} PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN} PUBLIC ${NEURUN_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN} PUBLIC ${CMAKE_SOURCE_DIR}/externals/tensorflow)
target_link_libraries(${LIB_NEURUN} arm_compute)
target_link_libraries(${LIB_NEURUN} tensorflow-lite)
target_link_libraries(${LIB_NEURUN} nnfw_util)
target_link_libraries(${LIB_NEURUN} nnfw_support_nnapi)

# TODO This should be optional
target_link_libraries(${LIB_NEURUN} ${LIB_NEURUN_BACKEND_CPU})
target_link_libraries(${LIB_NEURUN} ${LIB_NEURUN_BACKEND_ACL_CL})

target_compile_options(${LIB_NEURUN} PRIVATE -Wall -Wextra -Werror)

set_target_properties(${LIB_NEURUN} PROPERTIES OUTPUT_NAME neuralnetworks)

install(TARGETS ${LIB_NEURUN} DESTINATION lib/neurun)


# Unit Tests

set(TEST_NEURUN test_neurun)

file(GLOB_RECURSE TESTS "test/*.cc")

add_executable(${TEST_NEURUN} ${TESTS})
target_link_libraries(${TEST_NEURUN} ${LIB_NEURUN})
target_link_libraries(${TEST_NEURUN} gtest)
target_link_libraries(${TEST_NEURUN} gtest_main)
target_link_libraries(${TEST_NEURUN} ${LIB_PTHREAD})
add_test(${TEST_NEURUN} ${TEST_NEURUN})

install(TARGETS ${TEST_NEURUN} DESTINATION unittest)
