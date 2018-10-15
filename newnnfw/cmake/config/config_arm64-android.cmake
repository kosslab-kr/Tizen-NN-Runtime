set(ANDROID_STANDALONE $ENV{ROOTFS_ARM64})
set(CROSS_NDK_TOOLCHAIN ${ANDROID_STANDALONE}/bin)
set(CROSS_ROOTFS ${ANDROID_STANDALONE}/sysroot)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

## Specify the toolchain
set(TOOLCHAIN "aarch64-linux-android")
set(CMAKE_PREFIX_PATH ${CROSS_NDK_TOOLCHAIN})
set(TOOLCHAIN_PREFIX ${TOOLCHAIN}-)

find_program(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}clang)
find_program(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}clang++)
find_program(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}clang)
find_program(CMAKE_AR ${TOOLCHAIN_PREFIX}ar)
find_program(CMAKE_LD ${TOOLCHAIN_PREFIX}ar)
find_program(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy)
find_program(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}objdump)

add_compile_options(--sysroot=${CROSS_ROOTFS})
add_compile_options(-fPIE)

## Needed for Android or bionic specific conditionals
#add_compile_options(-D__ANDROID__)
#add_compile_options(-D__BIONIC__)

## NOTE Not sure this is safe. This may cause side effects.
## Without this, it cannot find `std::stol`, `std::stoi` and so on, with android toolchain
add_compile_options(-D_GLIBCXX_USE_C99=1)

set(CROSS_LINK_FLAGS "${CROSS_LINK_FLAGS} -B${CROSS_ROOTFS}/usr/lib/gcc/${TOOLCHAIN}")
set(CROSS_LINK_FLAGS "${CROSS_LINK_FLAGS} -L${CROSS_ROOTFS}/lib/${TOOLCHAIN}")
set(CROSS_LINK_FLAGS "${CROSS_LINK_FLAGS} --sysroot=${CROSS_ROOTFS}")
set(CROSS_LINK_FLAGS "${CROSS_LINK_FLAGS} -fPIE -pie")

set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${CROSS_LINK_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CROSS_LINK_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${CROSS_LINK_FLAGS}" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH "${CROSS_ROOTFS}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
