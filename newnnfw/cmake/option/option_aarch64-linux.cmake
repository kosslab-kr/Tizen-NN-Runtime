#
# aarch64 linux compile options
#

message(STATUS "Building for AARCH64 Linux")

# include linux common
include("cmake/option/option_linux.cmake")

if(NOT EXISTS "${ROOTFS_ARM64}/lib/aarch64-linux-gnu")
  message(FATAL_ERROR "Please prepare RootFS for ARM64")
endif()

# addition for aarch64-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    )
