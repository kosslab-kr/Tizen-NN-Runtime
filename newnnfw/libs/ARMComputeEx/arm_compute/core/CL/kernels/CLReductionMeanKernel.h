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
#ifndef __ARM_COMPUTE_CLREDUCTIONMEANKERNEL_H__
#define __ARM_COMPUTE_CLREDUCTIONMEANKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reduction operation kernel */
class CLReductionMeanKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLReductionMeanKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLReductionMeanKernel(const CLReductionMeanKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLReductionMeanKernel &operator=(const CLReductionMeanKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLReductionMeanKernel(CLReductionMeanKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLReductionMeanKernel &operator=(CLReductionMeanKernel &&) = default;
  /** Default destructor */
  ~CLReductionMeanKernel() = default;

  /** Set the input and output tensors.
   *
   * @param[in]  input  Source tensor. Data types supported: F32. Data layouts supported: NCHW.
   * @param[out] output Destination tensor. Data types and data layouts supported: Same as @p input.
   *                    Output will have the same number of dimensions as input.
   * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0, 1
   */
  void configure(const ICLTensor *input, ICLTensor *output, std::vector<uint32_t> axis);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLReductionMeanKernel.
   *
   * @param[in] input  Source tensor info. Data types supported: F32. Data layouts supported: NCHW.
   * @param[in] output Destination tensor info. Data types and data layouts supported: Same as @p
   * input.
   *                   Output will have the same number of dimensions as input.
   * @param[in] axis   Axis along which to reduce. Supported reduction axis : 0, 1
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         std::vector<uint32_t> axis);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;
  BorderSize border_size() const override;

private:
  const ICLTensor *_input;
  ICLTensor *_output;
  std::vector<uint32_t> _reduction_axis;
  BorderSize _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREDUCTIONMEANKERNEL_H__ */
