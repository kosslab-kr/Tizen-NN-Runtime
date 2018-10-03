#
# armv7l linux compile options
#

message(STATUS "Building for ARMv7l Linux")

# include linux common
include("cmake/option/option_linux.cmake")

if(NOT EXISTS "${ROOTFS_ARM}/lib/arm-linux-gnueabihf")
  message(FATAL_ERROR "Please prepare RootFS for ARM")
endif()

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mcpu=cortex-a7"
    "-mfloat-abi=hard"
    "-mfpu=neon-vfpv4"
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    )

# remove warning from arm cl
# https://github.com/ARM-software/ComputeLibrary/issues/330
set(GCC_VERSION_DISABLE_WARNING 6.0)
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER GCC_VERSION_DISABLE_WARNING)
  message(STATUS "GCC version higher than ${GCC_VERSION_DISABLE_WARNING}")
  set(FLAGS_CXXONLY ${FLAGS_CXXONLY}
      "-Wno-ignored-attributes"
      )
endif()
