file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nnfw_support_nnapi ${SOURCES})
set_property(TARGET nnfw_support_nnapi PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnfw_support_nnapi PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nnfw_support_nnapi static_nnfw_util)
