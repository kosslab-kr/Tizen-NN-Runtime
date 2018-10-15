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


inline Tensor4D tensor4D_from_vector_no_step(const Vector *vector, int dim_x, int dim_y, int dim_z, int dim_w)
{
    int stride_x = vector->stride_x;
    int stride_y = stride_x * dim_x;
    int stride_z = stride_y * dim_y;
    int stride_w = stride_z * dim_z;
    Tensor4D tensor =
    {
        .ptr                           = vector->ptr,
        .offset_first_element_in_bytes = vector->offset_first_element_in_bytes,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y,
        .stride_z                      = stride_z,
        .stride_w                      = stride_w,
    };
    return tensor;
}

/** Extracts a strided slice up to 4-dimensions
 *
 * @note Datatype should be given as a preprocessor argument using -DELEMENT_DATA_TYPE=type. e.g. -DELEMENT_DATA_TYPE=short
 * @note The size of an element should be given as a preprocessor argument using -DELEMENT_SIZE=size. e.g. -DELEMENT_SIZE=2
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  dims_in                              The 4-dimensional dimension of the input. Supported data types: S32
 * @param[in]  dims_out                             The 4-dimensional dimension of the output. Supported data types: S32
 * @param[in]  starts                               The stride of X dimension of input tensor to be sliced. Supported data types: S32
 * @param[in]  strides                              The stride of Y dimension of input tensor to be sliced. Supported data types: S32
 */
__kernel void strided_slice(VECTOR_DECLARATION(input),
                            VECTOR_DECLARATION(output),
                            const int4 dims_in,
                            const int4 dims_out,
                            const int4 starts,
                            const int4 strides)
{
    // TODO: Should be change to CONVERT_TO_TENSOR4D_STRUCT in order to reduce inference of the offset
    Vector vec_out = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output);
    Vector vec_in = CONVERT_TO_VECTOR_STRUCT_NO_STEP(input);

    // Implemenation
    // Infer a Tensor4D from output Vector and output's dimensions info
    // Infer a Tensor4D from input Vector and input's dimensions info
    // Infer indices of output as 4D from the offset of output vector
    // Infer indices of input as 4D from indices of output
    // out(offset of output vector) = in(offset of input)

    Tensor4D tensor_out = tensor4D_from_vector_no_step(&vec_out, dims_out.x, dims_out.y, dims_out.z, dims_out.w);
    Tensor4D tensor_in = tensor4D_from_vector_no_step(&vec_in, dims_in.x, dims_in.y, dims_in.z, dims_in.w);

    // Must be output_step_x == output_stride_x == an element's size
    const int offset_out = get_global_id(0) * output_stride_x;
    int4 indices_out =
    {
            get_global_id(0) % dims_out.x,
            (offset_out / tensor_out.stride_y) % dims_out.y,
            (offset_out / tensor_out.stride_z) % dims_out.z,
            (offset_out / tensor_out.stride_w) % dims_out.w,
    };

    int4 indices_in =
    {
            starts.x + (strides.x * indices_out.x),
            starts.y + (strides.y * indices_out.y),
            starts.z + (strides.z * indices_out.z),
            starts.w + (strides.w * indices_out.w),
    };

    *((__global ELEMENT_DATA_TYPE *)vector_offset(&vec_out, get_global_id(0))) = *((__global ELEMENT_DATA_TYPE *)tensor4D_offset(&tensor_in, indices_in.x, indices_in.y, indices_in.z, indices_in.w));
}
