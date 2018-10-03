# set host platform to build
if(NOT HOST_ARCH OR "${HOST_ARCH}" STREQUAL "")
  set(HOST_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

# set target platform to run
if(NOT TARGET_ARCH OR "${TARGET_ARCH}" STREQUAL "")
  set(TARGET_ARCH "${HOST_ARCH}")
endif()

if(NOT DEFINED TARGET_OS)
  set(TARGET_OS "${HOST_OS}")
endif()

if("${HOST_ARCH}" STREQUAL "x86_64")
  set(HOST_ARCH_BASE ${HOST_ARCH})
elseif("${HOST_ARCH}" STREQUAL "armv7l")
  set(HOST_ARCH_BASE "arm")
elseif("${HOST_ARCH}" STREQUAL "arm64")
  set(HOST_ARCH_BASE "arm64")
elseif("${HOST_ARCH}" STREQUAL "aarch64")
  set(HOST_ARCH_BASE "aarch64")
else()
  message(FATAL_ERROR "'${HOST_ARCH}' architecture is not supported")
endif()

if("${TARGET_ARCH}" STREQUAL "x86_64")
  set(TARGET_ARCH_BASE ${TARGET_ARCH})
elseif("${TARGET_ARCH}" STREQUAL "armv7l")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "arm64")
  set(TARGET_ARCH_BASE "arm64")
elseif("${TARGET_ARCH}" STREQUAL "aarch64")
  set(TARGET_ARCH_BASE "aarch64")
else()
  message(FATAL_ERROR "'${TARGET_ARCH}' architecture is not supported")
endif()

# Determine native or cross build
if("${HOST_ARCH}" STREQUAL "${TARGET_ARCH}")
  set(BUILD_IS_NATIVE True)
else()
  set(BUILD_IS_NATIVE False)
endif()

# host & target platform name
set(HOST_PLATFORM "${HOST_ARCH}-${HOST_OS}")
set(TARGET_PLATFORM "${TARGET_ARCH}-${TARGET_OS}")
