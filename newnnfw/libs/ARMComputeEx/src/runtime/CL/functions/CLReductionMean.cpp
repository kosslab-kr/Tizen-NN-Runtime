/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLReductionMean.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLReductionMeanKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLReductionMean::CLReductionMean() : _reduction_mean_kernel(), _fill_border_kernel() {}

Status CLReductionMean::validate(const ITensorInfo *input, const ITensorInfo *output,
                                 std::vector<uint32_t> axis)
{
  ARM_COMPUTE_RETURN_ON_ERROR(CLReductionMeanKernel::validate(input, output, axis));
  return Status{};
}

void CLReductionMean::configure(ICLTensor *input, ICLTensor *output, std::vector<uint32_t> axis)
{
  _reduction_mean_kernel.configure(input, output, axis);
  _fill_border_kernel.configure(input, _reduction_mean_kernel.border_size(), BorderMode::CONSTANT,
                                PixelValue(0));
}

void CLReductionMean::run()
{
  CLScheduler::get().enqueue(_fill_border_kernel);
  CLScheduler::get().enqueue(_reduction_mean_kernel);
}
