#
# config for arm-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# where is the target environment
set(ROOTFS_ARM $ENV{ROOTFS_ARM})
if(NOT EXISTS "${ROOTFS_ARM}/lib/arm-linux-gnueabihf")
  set(ROOTFS_ARM "${CMAKE_CURRENT_LIST_DIR}/../../tools/cross/rootfs/arm")
endif()

set(CMAKE_SYSROOT ${ROOTFS_ARM})
set(CMAKE_FIND_ROOT_PATH ${ROOTFS_ARM})
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${ROOTFS_ARM}"
    CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${ROOTFS_ARM}"
    CACHE INTERNAL "" FORCE)

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
