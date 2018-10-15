#
# armv7l tizen compile options
#

message(STATUS "Building for ARMv7l(softfp) Tizen")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("cmake/option/option_linux.cmake")

# TODO : support rootfs setting for tizen cross-build

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-mcpu=cortex-a8"
    "-mfloat-abi=softfp"
    "-mfpu=neon-vfpv4"
    "-funsafe-math-optimizations"
    "-ftree-vectorize"
    )
