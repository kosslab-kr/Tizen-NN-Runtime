/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2017 ARM Limited.
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

#if defined(WIDTH)
/** Perform reduce max
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types:  F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[out] output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[out] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[out] output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void reduce_max(VECTOR_DECLARATION(input),
                         VECTOR_DECLARATION(output))
{
    Vector input = CONVERT_TO_VECTOR_STRUCT(input);
    Vector output = CONVERT_TO_VECTOR_STRUCT(output);

    __global float *input_addr = (__global float *)(input.ptr);
    __global float *output_addr = (__global float *)(output.ptr);

    float max_value = *input_addr;
    for(int x = 1; x < WIDTH; x++)
    {
        float value = *(input_addr + x);
        max_value = max(value, max_value);
    }

    // Store max
    *output_addr = max_value;
}
#endif // defined(WIDTH)
