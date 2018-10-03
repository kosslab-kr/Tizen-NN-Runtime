file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_NEURUN_BACKEND_CPU} STATIC ${SOURCES})

target_include_directories(${LIB_NEURUN_BACKEND_CPU} PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_BACKEND_CPU} PUBLIC ${NEURUN_INCLUDE_DIR})
target_include_directories(${LIB_NEURUN_BACKEND_CPU} PUBLIC ${CMAKE_SOURCE_DIR}/externals/tensorflow)

target_link_libraries(${LIB_NEURUN_BACKEND_CPU} arm_compute) # TODO We should not need this
target_link_libraries(${LIB_NEURUN_BACKEND_CPU} tensorflow-lite)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU} nnfw_util)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU} nnfw_support_nnapi)
target_link_libraries(${LIB_NEURUN_BACKEND_CPU} ${LIB_NEURUN_KERNEL_CPU})

target_compile_options(${LIB_NEURUN_BACKEND_CPU} PRIVATE -Wall -Wextra -Werror)

set_target_properties(${LIB_NEURUN_BACKEND_CPU} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NEURUN_BACKEND_CPU} PROPERTIES OUTPUT_NAME backend_cpu)
install(TARGETS ${LIB_NEURUN_BACKEND_CPU} DESTINATION lib/neurun)
