/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2016-2018 ARM Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __ARM_COMPUTE_CLCAST_H__
#define __ARM_COMPUTE_CLCAST_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLCastKernel
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/S32/F16/F32.
 * @note The function converts the input tensor to the tensor of the output tensor's type.
 */
class CLCast : public ICLSimpleFunction
{
public:
  /** Initialise the kernel's input and output.
   *
   * @param[in, out] input Input tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   *                       The input tensor is [in, out] because its TensorInfo might be modified
   * inside the kernel.
   * @param[out]     output Output tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   */
  void configure(ICLTensor *input, ICLTensor *output);
};
}
#endif /* __ARM_COMPUTE_CLCAST_H__ */
