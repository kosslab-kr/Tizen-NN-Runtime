if(OBS_BUILD)
  enable_testing()
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIR})
  set(GTest_FOUND TRUE)
  return()
endif(OBS_BUILD)

if(${BUILD_GTEST})
  nnfw_include(ExternalSourceTools)
  nnfw_include(ExternalProjectTools)
  nnfw_include(OptionTools)

  envoption(GTEST_URL https://github.com/google/googletest/archive/release-1.8.0.zip)

  ExternalSource_Download("gtest" ${GTEST_URL})

  # gtest_SOURCE_DIR is used in gtest subdirectorty's cmake
  set(sourcedir_gtest ${gtest_SOURCE_DIR})
  unset(gtest_SOURCE_DIR)

  if(NOT TARGET gtest_main)
    add_extdirectory(${sourcedir_gtest} gtest)
  endif(NOT TARGET gtest_main)

  set(GTest_FOUND TRUE)
  return()
endif(${BUILD_GTEST})

### Find and use pre-installed Google Test
find_package(GTest)
find_package(Threads)

if(${GTEST_FOUND} AND TARGET Threads::Threads)
  if(NOT TARGET gtest)
    add_library(gtest INTERFACE)
    target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(gtest INTERFACE ${GTEST_LIBRARIES} Threads::Threads)
  endif(NOT TARGET gtest)

  if(NOT TARGET gtest_main)
    add_library(gtest_main INTERFACE)
    target_include_directories(gtest_main INTERFACE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(gtest_main INTERFACE gtest)
    target_link_libraries(gtest_main INTERFACE ${GTEST_MAIN_LIBRARIES})
  endif(NOT TARGET gtest_main)

  set(GTest_FOUND TRUE)
endif(${GTEST_FOUND} AND TARGET Threads::Threads)
