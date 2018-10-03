file(GLOB SOURCES "*.cc")

add_library(${LIB_NEURUN_KERNEL_CPU} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_KERNEL_CPU} PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_KERNEL_CPU} PUBLIC ${NEURUN_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_KERNEL_CPU} PUBLIC ${CMAKE_SOURCE_DIR}/externals/tensorflow)

target_link_libraries(${LIB_NEURUN_KERNEL_CPU} arm_compute) # TODO We should not need this
target_link_libraries(${LIB_NEURUN_KERNEL_CPU} tensorflow-lite)

set_target_properties(${LIB_NEURUN_KERNEL_CPU} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_KERNEL_CPU} PROPERTIES OUTPUT_NAME kernel_cpu)
install(TARGETS ${LIB_NEURUN_KERNEL_CPU} DESTINATION lib/neurun)
