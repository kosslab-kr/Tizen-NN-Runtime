#
# aarch64 tizen compile options
#

message(STATUS "Building for AARCH64 Tizen")

# TODO : add and use option_tizen if something uncommon comes up
# include linux common
include("cmake/option/option_linux.cmake")

# TODO : support rootfs setting for tizen cross-build

# addition for aarch64-tizen
set(FLAGS_COMMON ${FLAGS_COMMON}
    )
