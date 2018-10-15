###
### ARM Compute Library
###
set(ACL_BASE ${CMAKE_SOURCE_DIR}/externals/acl)
set(ACL_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/acl_generated)
set(ACL_VERSION_TAG "${ACL_GENERATED}/arm_compute_version.embed")

# Create 'arm_compute_version.embed'
add_custom_command(OUTPUT ${ACL_VERSION_TAG}
                   COMMAND mkdir -p "${ACL_GENERATED}"
                   COMMAND echo '"unknown"' > "${ACL_VERSION_TAG}")

file(GLOB_RECURSE ACL_UTIL_SRCS "${ACL_BASE}/src/core/utils/*.cpp")

### ARM Compute Library - Foundation library (such as I/O and logging)
if(BUILD_ACL_STATIC_LIB)
  add_library(acl_foundation ${ACL_UTIL_SRCS})
  target_include_directories(acl_foundation PUBLIC "${ACL_BASE}")
  target_include_directories(acl_foundation PUBLIC "${ACL_BASE}/include")
  target_link_libraries(acl_foundation dl ${LIB_PTHREAD})
endif(BUILD_ACL_STATIC_LIB)

###
### ARM Compute Library Common (Core & Runtime)
###
file(GLOB ACL_CORE_COMMON_SRCS "${ACL_BASE}/src/core/*.cpp")
list(APPEND ACL_CORE_COMMON_SRCS ${ACL_VERSION_TAG})
# Both CL & NEON runtime funtions use these CPP kernels
list(APPEND ACL_CORE_COMMON_SRCS "${ACL_BASE}/src/core/CPP/kernels/CPPCornerCandidatesKernel.cpp")
list(APPEND ACL_CORE_COMMON_SRCS "${ACL_BASE}/src/core/CPP/kernels/CPPDetectionWindowNonMaximaSuppressionKernel.cpp")
list(APPEND ACL_CORE_COMMON_SRCS "${ACL_BASE}/src/core/CPP/kernels/CPPSortEuclideanDistanceKernel.cpp")

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_core_common ${ACL_CORE_COMMON_SRCS})
  target_include_directories(acl_core_common PUBLIC "${ACL_GENERATED}")
  target_link_libraries(acl_core_common acl_foundation)
endif(BUILD_ACL_STATIC_LIB)

file(GLOB ACL_RUNTIME_COMMON_SRCS "${ACL_BASE}/src/runtime/*.cpp")
# src/runtime/Scheduler.cpp depends on this scheduler
list(APPEND ACL_RUNTIME_COMMON_SRCS "${ACL_BASE}/src/runtime/CPP/SingleThreadScheduler.cpp")

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_core_opencl ${ACL_CORE_OPENCL_SRCS})
  target_link_libraries(acl_core_opencl acl_core_common OpenCL)
endif(BUILD_ACL_STATIC_LIB)

###
### ARM Compute Library Open CL (Core & Runtime & Example)
###
file(GLOB ACL_CORE_OPENCL_SRCS "${ACL_BASE}/src/core/CL/*.cpp")
file(GLOB ACL_CORE_OPENCL_KERNEL_SRCS "${ACL_BASE}/src/core/CL/kernels/*.cpp")
list(APPEND ACL_CORE_OPENCL_SRCS ${ACL_CORE_OPENCL_KERNEL_SRCS})

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_runtime_opencl ${ACL_RUNTIME_OPENCL_SRCS})
  target_link_libraries(acl_runtime_opencl acl_runtime_common acl_core_opencl)
endif(BUILD_ACL_STATIC_LIB)

file(GLOB_RECURSE ACL_RUNTIME_OPENCL_SRCS "${ACL_BASE}/src/runtime/CL/*.cpp")

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_core_neon ${ACL_CORE_NEON_SRCS})
  target_include_directories(acl_core_neon PUBLIC "${ACL_BASE}/arm_compute/core/NEON/kernels/assembly")
  target_link_libraries(acl_core_neon acl_core_common)
endif(BUILD_ACL_STATIC_LIB)

###
### ARM Compute Library NEON (Core & Runtime & Example)
###
file(GLOB ACL_CORE_NEON_SRCS "${ACL_BASE}/src/core/NEON/kernels/*.cpp" "${ACL_BASE}/src/core/NEON/kernels/arm32/*.cpp")
file(GLOB_RECURSE ACL_CORE_NEON_CONVOLUTION_SRCS "${ACL_BASE}/src/core/NEON/kernels/convolution/winograd/*.cpp" "${ACL_BASE}/src/core/NEON/kernels/convolution/depthwise/*.cpp")
list(APPEND ACL_CORE_NEON_SRCS ${ACL_CORE_NEON_CONVOLUTION_SRCS})
list(APPEND ACL_CORE_NEON_SRCS "${ACL_BASE}/src/core/CPP/ICPPSimpleKernel.cpp")
list(APPEND ACL_CORE_NEON_SRCS "${ACL_BASE}/src/core/CPP/kernels/CPPPermuteKernel.cpp")

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_runtime_neon ${ACL_RUNTIME_NEON_SRCS})
  target_link_libraries(acl_runtime_neon acl_runtime_common acl_core_neon)
endif(BUILD_ACL_STATIC_LIB)

file(GLOB_RECURSE ACL_RUNTIME_NEON_SRCS "${ACL_BASE}/src/runtime/NEON/*.cpp")
# runtime/NEON/functions/NEWinogradLayer.h use this implementation
list(APPEND ACL_RUNTIME_NEON_SRCS "${ACL_BASE}/src/runtime/CPP/ICPPSimpleFunction.cpp")
list(APPEND ACL_RUNTIME_NEON_SRCS "${ACL_BASE}/src/runtime/CPP/functions/CPPPermute.cpp")

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_graph ${ACL_GRAPH_SRCS})
  target_link_libraries(acl_graph acl_runtime_opencl acl_runtime_neon)
endif(BUILD_ACL_STATIC_LIB)

# TODO Support Open MP core(?)
# TODO Support Open GLES core(?)

###
### ARM Compute Library (Graph & Example)
###
file(GLOB ACL_GRAPH_COMMON_SRCS "${ACL_BASE}/src/graph/*.cpp" "${ACL_BASE}/src/graph/nodes/*.cpp")
file(GLOB ACL_GRAPH_OPENCL_SRCS "${ACL_BASE}/src/graph/CL/*.cpp" "${ACL_BASE}/src/graph/operations/CL*.cpp")
file(GLOB ACL_GRAPH_NEON_SRCS "${ACL_BASE}/src/graph/NE/*.cpp" "${ACL_BASE}/src/graph/operations/NE*.cpp")

list(APPEND ACL_GRAPH_SRCS ${ACL_GRAPH_COMMON_SRCS})
list(APPEND ACL_GRAPH_SRCS ${ACL_GRAPH_OPENCL_SRCS})
list(APPEND ACL_GRAPH_SRCS ${ACL_GRAPH_NEON_SRCS})

if(BUILD_ACL_STATIC_LIB)
  add_library(acl_graph ${ACL_GRAPH_SRCS})
  target_link_libraries(acl_graph acl_runtime_opencl acl_runtime_neon)
endif(BUILD_ACL_STATIC_LIB)

###
### ARM Compute Shared Libraries
###
list(APPEND ACL_CORE_SRCS ${ACL_UTIL_SRCS})
list(APPEND ACL_CORE_SRCS ${ACL_CORE_COMMON_SRCS})
list(APPEND ACL_CORE_SRCS ${ACL_CORE_OPENCL_SRCS})
list(APPEND ACL_CORE_SRCS ${ACL_CORE_NEON_SRCS})

add_library(arm_compute_core SHARED ${ACL_CORE_SRCS})
target_include_directories(arm_compute_core PUBLIC "${ACL_GENERATED}")
target_include_directories(arm_compute_core PUBLIC "${ACL_BASE}")
target_include_directories(arm_compute_core PUBLIC "${ACL_BASE}/include")
target_include_directories(arm_compute_core PUBLIC "${ACL_BASE}/arm_compute/core/NEON/kernels/assembly")
target_link_libraries(arm_compute_core dl ${LIB_PTHREAD})
install(TARGETS arm_compute_core DESTINATION lib)

list(APPEND ACL_RUNTIME_SRCS ${ACL_RUNTIME_COMMON_SRCS})
list(APPEND ACL_RUNTIME_SRCS ${ACL_RUNTIME_OPENCL_SRCS})
list(APPEND ACL_RUNTIME_SRCS ${ACL_RUNTIME_NEON_SRCS})

add_library(arm_compute SHARED ${ACL_RUNTIME_SRCS})
target_link_libraries(arm_compute arm_compute_core OpenCL)
install(TARGETS arm_compute DESTINATION lib)

add_library(arm_compute_graph SHARED ${ACL_GRAPH_SRCS})
target_link_libraries(arm_compute_graph arm_compute)
install(TARGETS arm_compute_graph DESTINATION lib)

add_library(arm_compute_test SHARED "${ACL_BASE}/utils/Utils.cpp")
target_link_libraries(arm_compute_test arm_compute)

add_library(arm_compute_graph_test SHARED "${ACL_BASE}/utils/GraphUtils.cpp")
target_link_libraries(arm_compute_graph_test arm_compute_graph arm_compute_test)

add_executable(cl_convolution "${ACL_BASE}/examples/cl_convolution.cpp")
target_compile_definitions(cl_convolution PRIVATE ARM_COMPUTE_CL)
target_link_libraries(cl_convolution arm_compute_test)

add_executable(neon_convolution "${ACL_BASE}/examples/neon_convolution.cpp")
target_link_libraries(neon_convolution arm_compute_test)

add_executable(graph_lenet "${ACL_BASE}/examples/graph_lenet.cpp")
target_link_libraries(graph_lenet arm_compute_graph_test)
