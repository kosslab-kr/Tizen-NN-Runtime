if(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")
  return()
endif(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")

list(APPEND OPENCL_INFO_SOURCE "src/opencl_info.cc")

add_executable(opencl_info ${OPENCL_INFO_SOURCE})
target_include_directories(opencl_info PUBLIC ${CMAKE_SOURCE_DIR}/externals/acl)
target_include_directories(opencl_info PUBLIC ${CMAKE_SOURCE_DIR}/externals/acl/include)
target_link_libraries(opencl_info arm_compute)

install(TARGETS opencl_info DESTINATION bin)
