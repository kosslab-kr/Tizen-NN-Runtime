file(GLOB_RECURSE NNAPI_LOGGING_SRCS "src/*.cc")

add_library(neuralnetworks SHARED ${NNAPI_LOGGING_SRCS})
target_include_directories(neuralnetworks PUBLIC ${NNFW_INCLUDE_DIR})
target_include_directories(neuralnetworks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
