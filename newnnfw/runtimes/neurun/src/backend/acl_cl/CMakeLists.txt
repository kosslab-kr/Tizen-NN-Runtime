file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_NEURUN_BACKEND_ACL_CL} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_BACKEND_ACL_CL} PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_BACKEND_ACL_CL} PUBLIC ${NEURUN_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_BACKEND_ACL_CL} PUBLIC ${CMAKE_SOURCE_DIR}/externals/tensorflow) # TODO Remove this file. We should not need this.

target_link_libraries(${LIB_NEURUN_BACKEND_ACL_CL} arm_compute)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_CL} nnfw_support_nnapi)
target_link_libraries(${LIB_NEURUN_BACKEND_ACL_CL} ${LIB_NEURUN_KERNEL_ACL_CL})

target_compile_options(${LIB_NEURUN_BACKEND_ACL_CL} PRIVATE -Wall -Wextra -Werror)

set_target_properties(${LIB_NEURUN_BACKEND_ACL_CL} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_BACKEND_ACL_CL} PROPERTIES OUTPUT_NAME backend_acl_cl)
install(TARGETS ${LIB_NEURUN_BACKEND_ACL_CL} DESTINATION lib/neurun)
