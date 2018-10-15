include("cmake/option/option_linux.cmake")

# On Android, pthread is contained in bionic(libc)
set(LIB_PTHREAD "")

# SIMD for arm64
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-ftree-vectorize"
    )
