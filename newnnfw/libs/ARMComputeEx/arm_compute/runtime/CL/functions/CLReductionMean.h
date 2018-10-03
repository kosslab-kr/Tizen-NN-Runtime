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

#ifndef __ARM_COMPUTE_CLREDUCTIONMEAN_H__
#define __ARM_COMPUTE_CLREDUCTIONMEAN_H__

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLReductionMeanKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class ICLTensor;

/** Perform reduction operation.
 */
class CLReductionMean : public IFunction
{
public:
  /** Default Constructor.
   */
  CLReductionMean();

  /** Set the input and output tensors.
   *
   * @param[in]  input  Source tensor. Data types supported: F32. Data layouts supported: NCHW.
   * @param[out] output Destination tensor. Data types and data layouts supported: Same as @p input.
   * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0,1
   */
  void configure(ICLTensor *input, ICLTensor *output, std::vector<uint32_t> axis);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLReductionMean.
   *
   * @param[in] input  Source tensor info. Data types supported: F32. Data layouts supported: NCHW.
   * @param[in] output Destination tensor info. Data types and data layouts supported: Same as @p
   * input.
   * @param[in] axis   Axis along which to reduce. Supported reduction axis : 0,1
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         std::vector<uint32_t> axis);

  // Inherited methods overridden:
  void run() override;

private:
  CLReductionMeanKernel _reduction_mean_kernel;
  CLFillBorderKernel _fill_border_kernel;
};
}
#endif /*__ARM_COMPUTE_CLREDUCTIONMEAN_H__ */
