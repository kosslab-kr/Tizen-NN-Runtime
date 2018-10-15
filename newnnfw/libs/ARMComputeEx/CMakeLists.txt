if("${TARGET_ARCH}" STREQUAL "x86_64")
  return()
endif()

nnfw_find_package(ARMCompute REQUIRED)

set(ACL_EX_BASE ${CMAKE_SOURCE_DIR}/libs/ARMComputeEx)

file(GLOB_RECURSE ACL_EX_SRCS "${ACL_EX_BASE}/*.cpp")

# generate embeded cl_kernel
execute_process (
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/libs/ARMComputeEx"
    COMMAND bash -c "python resolve_includes.py"
)

add_library(arm_compute_ex SHARED ${ACL_EX_SRCS})
set_target_properties(arm_compute_ex PROPERTIES COMPILE_FLAGS "-DEMBEDDED_KERNELS=1")
target_include_directories(arm_compute_ex PUBLIC ${CMAKE_SOURCE_DIR}/libs/ARMComputeEx)
target_link_libraries(arm_compute_ex arm_compute_core)
install(TARGETS arm_compute_ex DESTINATION lib)
