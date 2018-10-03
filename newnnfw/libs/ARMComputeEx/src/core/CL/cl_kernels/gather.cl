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

/** Perform gather
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  input1_ptr                            Pointer to the first source tensor. Supported data types: U8/S32/F32
 * @param[in]  input1_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input1_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input1_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input1_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input1_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input1_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input1_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  input2_ptr                            Pointer to the first source tensor. Supported data types: U32
 * @param[in]  input2_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input2_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input2_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gather(IMAGE_DECLARATION(input1),
                    VECTOR_DECLARATION(input2),
                    IMAGE_DECLARATION(output))
{
    Image in1  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(input1);
    Vector in2  = CONVERT_TO_VECTOR_STRUCT(input2);
    Image out = CONVERT_TO_IMAGE_STRUCT_NO_STEP(output);

    VEC_DATA_TYPE(DATA_TYPE_IN2, 2)
    in2_data = CONVERT(vload2(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(DATA_TYPE_IN2, 2));

    //TODO: performance tuning for memcopy
    int index = in2_data.s0;
    int stride=input1_stride_y/input1_stride_x;

    for(int i=0; i<stride; i++){
        *((__global DATA_TYPE_OUT *)offset(&out, i,get_global_id(0)))=*((__global DATA_TYPE_IN1 *)offset(&in1, i,index));
    }
}

__kernel void gather_1d_out(IMAGE_DECLARATION(input1),
                    VECTOR_DECLARATION(input2),
                    VECTOR_DECLARATION(output))
{
    Image in1  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(input1);
    Vector in2  = CONVERT_TO_VECTOR_STRUCT(input2);
    Vector out = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output);

    VEC_DATA_TYPE(DATA_TYPE_IN2, 2)
    in2_data = CONVERT(vload2(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(DATA_TYPE_IN2, 2));

    //TODO: performance tuning for memcopy
    int index = in2_data.s0;
    int stride=input1_stride_y/input1_stride_x;

    for(int i=0; i<stride; i++){
        *((__global DATA_TYPE_OUT *)vector_offset(&out, i+get_global_id(0)))=*((__global DATA_TYPE_IN1 *)offset(&in1, i, index));
    }
}

__kernel void gather_1d(VECTOR_DECLARATION(input1),
                    VECTOR_DECLARATION(input2),
                    VECTOR_DECLARATION(output))
{
    Vector in1  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(input1);
    Vector in2  = CONVERT_TO_VECTOR_STRUCT(input2);
    Vector out = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output);

    VEC_DATA_TYPE(DATA_TYPE_IN2, 2)
    in2_data = CONVERT(vload2(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(DATA_TYPE_IN2, 2));

    //TODO: performance tuning for memcopy
    int index = in2_data.s0;
    *((__global DATA_TYPE_OUT *)vector_offset(&out, get_global_id(0)))=*((__global DATA_TYPE_IN1 *)vector_offset(&in1, index));
}
