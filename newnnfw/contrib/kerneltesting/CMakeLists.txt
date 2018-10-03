if(NOT BUILD_LABS)
  return()
endif(NOT BUILD_LABS)

if(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")
  return()
endif(NOT ${TARGET_ARCH_BASE} STREQUAL "arm")

nnfw_find_package(ARMCompute REQUIRED)

function(add_kerneltesting TESTNAME SRC_FILES)
    link_directories(${CMAKE_INSTALL_PREFIX}/lib)
    add_executable(${TESTNAME} ${SRC_FILES})
    target_include_directories(${TESTNAME} PUBLIC
                               ${NNFW_INCLUDE_DIR})
    target_link_libraries(${TESTNAME} nnfw_util arm_compute_graph)
    install(TARGETS ${TESTNAME} DESTINATION bin)
endfunction()

# TODO: Enable conv2d on Tizen
if (NOT ${TARGET_OS} STREQUAL "tizen")
  add_subdirectory(conv2d)
endif()
