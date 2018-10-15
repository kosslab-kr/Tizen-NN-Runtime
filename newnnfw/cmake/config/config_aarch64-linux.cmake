#
# config for aarch64-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# where is the target environment
set(ROOTFS_ARM64 $ENV{ROOTFS_ARM64})
if(NOT EXISTS "${ROOTFS_ARM64}/lib/aarch64-linux-gnu")
  set(ROOTFS_ARM64 "${CMAKE_CURRENT_LIST_DIR}/../../tools/cross/rootfs/arm64")
endif()

set(CMAKE_SYSROOT ${ROOTFS_ARM64})
set(CMAKE_FIND_ROOT_PATH ${ROOTFS_ARM64})
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${ROOTFS_ARM64}"
    CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${ROOTFS_ARM64}"
    CACHE INTERNAL "" FORCE)

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
