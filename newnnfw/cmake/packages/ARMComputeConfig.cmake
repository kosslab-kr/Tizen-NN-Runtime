function(_ARMCompute_Build)
  if(TARGET arm_compute_core)
    set(ARMCompute_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET arm_compute_core)

  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/ARMCompute" "${CMAKE_BINARY_DIR}/externals/ARMCompute")
  set(ARMCompute_FOUND TRUE PARENT_SCOPE)
endfunction(_ARMCompute_Build)

function(_ARMCompute_Import)
  include(FindPackageHandleStandardArgs)

  list(APPEND ARMCompute_INCLUDE_SEARCH_PATHS /usr/include)

  list(APPEND ARMCompute_LIB_SEARCH_PATHS /usr/lib)

  find_path(INCLUDE_DIR NAMES arm_compute/core/ITensor.h PATHS ${ARMCompute_INCLUDE_SEARCH_PATHS})

  find_library(CORE_LIBRARY NAMES  	 arm_compute_core  PATHS ${ARMCompute_LIB_SEARCH_PATHS})
  find_library(RUNTIME_LIBRARY NAMES arm_compute       PATHS ${ARMCompute_LIB_SEARCH_PATHS})
  find_library(GRAPH_LIBRARY NAMES   arm_compute_graph PATHS ${ARMCompute_LIB_SEARCH_PATHS})

  if(NOT INCLUDE_DIR)
    set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/externals/acl ${CMAKE_SOURCE_DIR}/externals/acl/include)
  endif(NOT INCLUDE_DIR)

  # NOTE '${CMAKE_INSTALL_PREFIX}/lib' should be searched as CI server places
  #       pre-built ARM compute libraries on this directory
  if(NOT CORE_LIBRARY AND EXISTS ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute_core.so)
    set(CORE_LIBRARY ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute_core.so)
  endif()

  if(NOT CORE_LIBRARY)
    return()
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
  endif()

  if(NOT RUNTIME_LIBRARY AND EXISTS ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute.so)
    set(RUNTIME_LIBRARY ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute.so)
  endif()

  if(NOT RUNTIME_LIBRARY)
    return()
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
  endif()

  if(NOT GRAPH_LIBRARY AND EXISTS ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute_graph.so)
    set(GRAPH_LIBRARY ${CMAKE_INSTALL_PREFIX}/lib/libarm_compute_graph.so)
  endif()

  if(NOT GRAPH_LIBRARY)
    return()
    set(ARMCompute_FOUND FALSE PARENT_SCOPE)
  endif()

  if(NOT TARGET arm_compute_core)
    add_library(arm_compute_core INTERFACE)
    target_include_directories(arm_compute_core INTERFACE ${INCLUDE_DIR})
	target_link_libraries(arm_compute_core INTERFACE dl ${LIB_PTHREAD})
    target_link_libraries(arm_compute_core INTERFACE ${CORE_LIBRARY})
    if (${TARGET_OS} STREQUAL "tizen")
      target_link_libraries(arm_compute_core INTERFACE OpenCL)
    endif()
  endif(NOT TARGET arm_compute_core)

  if(NOT TARGET arm_compute)
    add_library(arm_compute INTERFACE)
    target_include_directories(arm_compute INTERFACE ${INCLUDE_DIR})
    target_link_libraries(arm_compute INTERFACE ${RUNTIME_LIBRARY})
    target_link_libraries(arm_compute INTERFACE arm_compute_core)
  endif(NOT TARGET arm_compute)

  if(NOT TARGET arm_compute_graph)
    add_library(arm_compute_graph INTERFACE)
    target_include_directories(arm_compute_graph INTERFACE ${INCLUDE_DIR})
    target_link_libraries(arm_compute_graph INTERFACE ${GRAPH_LIBRARY})
    target_link_libraries(arm_compute_graph INTERFACE arm_compute)
  endif(NOT TARGET arm_compute_graph)

  set(ARMCompute_FOUND TRUE PARENT_SCOPE)
endfunction(_ARMCompute_Import)

if(BUILD_ACL)
  _ARMCompute_Build()
else(BUILD_ACL)
  _ARMCompute_Import()
endif(BUILD_ACL)
