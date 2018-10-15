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
#ifndef __ARM_COMPUTE_CLCASTKERNEL_H__
#define __ARM_COMPUTE_CLCASTKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a cast operation */
class CLCastKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLCastKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLCastKernel(const CLCastKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLCastKernel &operator=(const CLCastKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLCastKernel(CLCastKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLCastKernel &operator=(CLCastKernel &&) = default;
  /** Default destructor */
  ~CLCastKernel() = default;
  /** Initialise the kernel's input and output.
   *
   * @param[in]  input  Input tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   * @param[in]  output Output tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   */
  void configure(const ICLTensor *input, ICLTensor *output);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input; /**< Source tensor */
  ICLTensor *_output;      /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCASTKERNEL_H__ */
