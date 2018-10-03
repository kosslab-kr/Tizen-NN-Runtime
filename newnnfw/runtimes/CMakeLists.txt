set(BUILD_RUNTIME_TESTS OFF)

if(BUILD_NEURUN)
  set(LIB_NEURUN neurun)
  add_subdirectory(neurun)
endif(BUILD_NEURUN)

if(BUILD_PURE_ARM_COMPUTE)
  set(LIB_PUREACL_RUNTIME nnapi_pure_arm_compute)
  add_subdirectory(pure_arm_compute)
  set(BUILD_RUNTIME_TESTS ON)
endif()

if(BUILD_RUNTIME_TESTS)
  add_subdirectory(tests)
endif(BUILD_RUNTIME_TESTS)

if(NOT OBS_BUILD)
  add_subdirectory(template)
  add_subdirectory(logging)
endif(NOT OBS_BUILD)
