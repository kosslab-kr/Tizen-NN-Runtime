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
#ifndef __ARM_COMPUTE_CLGATHERKERNEL_H__
#define __ARM_COMPUTE_CLGATHERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the gather kernel.
 *
 */
class CLGatherKernel : public ICLKernel
{
public:
  /** Default constructor.*/
  CLGatherKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLGatherKernel(const CLGatherKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLGatherKernel &operator=(const CLGatherKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLGatherKernel(CLGatherKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLGatherKernel &operator=(CLGatherKernel &&) = default;
  /** Initialise the kernel's input, output and border mode.
   *
   * @param[in]  input1          An input tensor. Data types supported: U8/S32/F32.
   * @param[in]  input2          An input tensor. Data types supported: S32.
   * @param[out] output          The output tensor, Data types supported: same as @p input1.
   */
  void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLGatherKernel
   *
   * @param[in]  input1          An input tensor. Data types supported: U8/S32/F32.
   * @param[in]  input2          An input tensor. Data types supported: S32.
   * @param[out] output          The output tensor, Data types supported: same as @p input1.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input1, const ITensorInfo *input2,
                         const ITensorInfo *output);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input1;
  const ICLTensor *_input2;
  ICLTensor *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGATHERKERNEL_H__ */
