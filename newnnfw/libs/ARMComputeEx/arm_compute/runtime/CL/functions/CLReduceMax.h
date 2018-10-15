/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLREDUCE_MAX_H__
#define __ARM_COMPUTE_CLREDUCE_MAX_H__

#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute TopK operation. This function calls the following OpenCL kernels:
 *
 * -# @ref CLTopKV2Kernel
 */
class CLReduceMax : public IFunction
{
public:
  /** Constructor */
  CLReduceMax();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLReduceMax(const CLReduceMax &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLReduceMax &operator=(const CLReduceMax &) = delete;
  /** Allow instances of this class to be moved */
  CLReduceMax(CLReduceMax &&) = default;
  /** Allow instances of this class to be moved */
  CLReduceMax &operator=(CLReduceMax &&) = default;
  /** Initialise the kernel's inputs and outputs.
   *
   * @note When locations of min and max occurrences are requested, the reported number of locations
   * is limited to the given array size.
   *
   * @param[in]  input     Input image. Data types supported: F32
   * @param[in]  axis      Axis to reduce. Data type supported: S32
   * @param[out] output    indices related to top k values. Data types supported: F32.
   */
  void configure(ICLTensor *input, int32_t axis, ICLTensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLPixelWiseDivision
   *
   * @param[in]  input     Input image. Data types supported: F32
   * @param[in]  axis      Axis to reduce. Data type supported: S32
   * @param[out] output    indices related to top k values. Data types supported: F32.     *
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, int32_t axis, const ITensorInfo *output);

  // Inherited methods overridden:
  void run() override;

private:
  void run_on_cpu();

  int32_t _axis;

  ICLTensor *_input;
  ICLTensor *_output;

  std::unique_ptr<ICLKernel> _kernel;
};
}
#endif /*__ARM_COMPUTE_CLREDUCE_MAX_H__ */
