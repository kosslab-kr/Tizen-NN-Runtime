set(KERNELTESTING_CONV2D kerneltesting_conv2d)

set(KERNELTESTING_CONV2D_SRCS "nnfw_conv2d_test.cpp"
                              "io_accessor.cpp")

set(GEMLOWP_INCUDE ${TFLITE_DEPEND_DIR}/gemmlowp/public)
set(EIGN_INCLUDE ${TFLITE_DEPEND_DIR}/eigen
                 ${TFLITE_DEPEND_DIR}/eigen/Eigen)

add_kerneltesting(${KERNELTESTING_CONV2D} "${KERNELTESTING_CONV2D_SRCS}")

target_include_directories(${KERNELTESTING_CONV2D} PUBLIC
                           ${GEMLOWP_INCUDE}
                           ${EIGN_INCLUDE}
                          )
