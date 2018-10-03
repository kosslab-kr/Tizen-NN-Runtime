file(GLOB SOURCES "*.cc")

add_library(${LIB_NEURUN_KERNEL_ACL_CL} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_KERNEL_ACL_CL} PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_KERNEL_ACL_CL} PUBLIC ${NEURUN_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_KERNEL_ACL_CL} PUBLIC ${CMAKE_SOURCE_DIR}/externals/tensorflow) # TODO We should not need this

target_link_libraries(${LIB_NEURUN_KERNEL_ACL_CL} arm_compute)
target_link_libraries(${LIB_NEURUN_KERNEL_ACL_CL} tensorflow-lite) # TODO We should not need this
target_link_libraries(${LIB_NEURUN_KERNEL_ACL_CL} ${LIB_NEURUN_KERNEL_CPU}) # TODO We should not need this

set_target_properties(${LIB_NEURUN_KERNEL_ACL_CL} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_KERNEL_ACL_CL} PROPERTIES OUTPUT_NAME kernel_acl_cl)
install(TARGETS ${LIB_NEURUN_KERNEL_ACL_CL} DESTINATION lib/neurun)
