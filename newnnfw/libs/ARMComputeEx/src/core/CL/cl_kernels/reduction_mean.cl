/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2016, 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "helpers.h"

inline DATA_TYPE sum_8(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 8)
    in = vload8(0, input);
    in.s0123 += in.s4567;
    in.s01 += in.s23;
    return ((in.s0 + in.s1));
}

/** This function calculates the sum and sum of squares of a given input image.
 *
 * @note To enable calculation sum of squares -DSTDDEV should be passed as a preprocessor argument.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] local_sum                         Local sum of all elements
 * @param[in]  height                            Height of the input image
 * @param[in]  divider                           Divider to calculate mean
 */
__kernel void reduction_mean(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    __local DATA_TYPE *local_sums, 
    int height,
    int divider)
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    float8 tmp_sum = 0;
    // Calculate partial sum

    for(int i = 0; i < height; i++)
    {
        local_sums[0] += sum_8((__global DATA_TYPE *)offset(&src, 0, i));
    }
    ((__global DATA_TYPE *)offset(&dst, get_global_id(0), get_global_id(1)))[0] = local_sums[0]/divider;
}
