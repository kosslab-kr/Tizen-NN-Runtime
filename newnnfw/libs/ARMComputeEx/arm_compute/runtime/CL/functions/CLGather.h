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
#ifndef __ARM_COMPUTE_CLGATHER_H__
#define __ARM_COMPUTE_CLGATHER_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLGatherKernel. */
class CLGather : public ICLSimpleFunction
{
public:
  /** Initialise the kernel's inputs, output and convertion policy.
   *
   * @param[in] input1          An input tensor. Data types supported: U8/S32/F32.
   * @param[in] input2          An indexes tensor. Data types supported: S32.
   * @param[out]     output          The output tensor, Data types supported: same as @p input1.
   */
  void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref CLGather
   *
   * @param[in] input1          An input tensor. Data types supported: U8/S32/F32.
   * @param[in] input2          An indexes tensor. Data types supported: S32.
   * @param[out]     output          The output tensor, Data types supported: same as @p input1.
   * @return a status
   */
  static Status validate(const ITensorInfo *input1, const ITensorInfo *input2,
                         const ITensorInfo *output);
};
}
#endif /*__ARM_COMPUTE_CLGATHER_H__ */
