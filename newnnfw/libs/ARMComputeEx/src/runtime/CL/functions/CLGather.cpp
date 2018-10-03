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
#include "arm_compute/runtime/CL/functions/CLGather.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGatherKernel.h"
#include "support/ToolchainSupport.h"

#include <utility>

using namespace arm_compute;

void CLGather::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
  auto k = arm_compute::support::cpp14::make_unique<CLGatherKernel>();
  k->configure(input1, input2, output);
  _kernel = std::move(k);
}

Status CLGather::validate(const ITensorInfo *input1, const ITensorInfo *input2,
                          const ITensorInfo *output)
{
  return CLGatherKernel::validate(input1, input2, output);
}
