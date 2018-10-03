if(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")
  return()
endif(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")

list(APPEND OPENCL_INFO_SOURCE "src/opencl_test.cc")

add_executable(opencl_test ${OPENCL_INFO_SOURCE})
target_include_directories(opencl_test PUBLIC ${CMAKE_SOURCE_DIR}/externals/acl)
target_include_directories(opencl_test PUBLIC ${CMAKE_SOURCE_DIR}/externals/acl/include)
target_link_libraries(opencl_test arm_compute)

