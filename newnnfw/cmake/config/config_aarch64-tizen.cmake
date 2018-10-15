#
# config for aarch64-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc-5)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-5)

set(TIZEN_TOOLCHAIN "aarch64-tizen-linux-gnu/6.2.1")

# where is the target environment
set(ROOTFS_ARM64 $ENV{ROOTFS_ARM64})
if(NOT EXISTS "${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
  set(ROOTFS_ARM64 "${CMAKE_SOURCE_DIR}/tools/cross/rootfs/arm64")
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

add_compile_options(--sysroot=${ROOTFS_ARM64})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${ROOTFS_ARM64}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${ROOTFS_ARM64}")

include_directories(SYSTEM ${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}/include/c++/)
include_directories(SYSTEM ${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}/include/c++/aarch64-tizen-linux-gnu)
add_compile_options(-Wno-deprecated-declarations) # compile-time option
add_compile_options(-D__extern_always_inline=inline) # compile-time option

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -B${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM64}/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM64}/usr/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -B${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM64}/lib64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM64}/usr/lib64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM64}/usr/lib64/gcc/${TIZEN_TOOLCHAIN}")
