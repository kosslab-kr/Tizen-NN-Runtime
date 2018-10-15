if(NOT BUILD_LABS)
  return()
endif(NOT BUILD_LABS)

if(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")
  return()
endif(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")

nnfw_find_package(ARMCompute REQUIRED)

file(GLOB_RECURSE NNAPI_BINDACL_SRCS "src/*.cc")

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(exp_bindacl SHARED ${NNAPI_BINDACL_SRCS})
target_include_directories(exp_bindacl PUBLIC ${NNFW_INCLUDE_DIR})
target_link_libraries(exp_bindacl nnfw_util arm_compute_graph)

# we need the library name to be 'neuralnetworks' and this will do the trick
set_target_properties(exp_bindacl PROPERTIES OUTPUT_NAME neuralnetworks)
