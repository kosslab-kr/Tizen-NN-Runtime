#
# config for arm-linux
#
include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER   arm-linux-gnueabi-gcc-5)
set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++-5)

set(TIZEN_TOOLCHAIN "armv7l-tizen-linux-gnueabi/6.2.1")

# where is the target environment
set(ROOTFS_ARM $ENV{ROOTFS_ARM})
if(NOT EXISTS "${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
  set(ROOTFS_ARM "${CMAKE_SOURCE_DIR}/tools/cross/rootfs/armel")
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



add_compile_options(-mthumb)
add_compile_options(-mfpu=neon-vfpv4)
add_compile_options(-mfloat-abi=softfp)
add_compile_options(--sysroot=${ROOTFS_ARM})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${ROOTFS_ARM}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${ROOTFS_ARM}")

include_directories(SYSTEM ${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}/include/c++/)
include_directories(SYSTEM ${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}/include/c++/armv7l-tizen-linux-gnueabi)
add_compile_options(-Wno-deprecated-declarations) # compile-time option
add_compile_options(-D__extern_always_inline=inline) # compile-time option

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -B${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM}/usr/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -B${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM}/usr/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTFS_ARM}/usr/lib/gcc/${TIZEN_TOOLCHAIN}")
