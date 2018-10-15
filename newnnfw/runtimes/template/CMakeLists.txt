file(GLOB_RECURSE SOURCES "src/*.cc")

add_library(nnapi_template SHARED ${SOURCES})
target_include_directories(nnapi_template PUBLIC ${NNFW_INCLUDE_DIR})
set_target_properties(nnapi_template PROPERTIES OUTPUT_NAME neuralnetworks)
