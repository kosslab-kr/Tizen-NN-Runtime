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
#include "arm_compute/runtime/CL/functions/CLPixelWiseDivision.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseDivisionKernel.h"
#include "support/ToolchainSupport.h"

#include <utility>

using namespace arm_compute;

void CLPixelWiseDivision::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output,
                                    float scale, ConvertPolicy overflow_policy,
                                    RoundingPolicy rounding_policy)
{
  auto k = arm_compute::support::cpp14::make_unique<CLPixelWiseDivisionKernel>();
  k->configure(input1, input2, output, scale, overflow_policy, rounding_policy);
  _kernel = std::move(k);

  if (output->info()->dimension(0) > 1)
  {
    ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

    if (broadcasted_info->info()->dimension(0) == 1)
    {
      _border_handler.configure(broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
    }
  }
}

Status CLPixelWiseDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2,
                                     const ITensorInfo *output, float scale,
                                     ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
  return CLPixelWiseDivisionKernel::validate(input1, input2, output, scale, overflow_policy,
                                             rounding_policy);
}
