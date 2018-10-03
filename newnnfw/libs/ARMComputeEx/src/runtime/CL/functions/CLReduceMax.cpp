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
#include "arm_compute/runtime/CL/functions/CLReduceMax.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "support/ToolchainSupport.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/kernels/CLReduceMaxKernel.h"

#include <vector>
#include <algorithm>

#include <utility>

#define REDUCE_MAX_RUN_ON_CPU 1

namespace arm_compute
{

CLReduceMax::CLReduceMax() : _axis(0), _input(nullptr), _output(nullptr), _kernel(nullptr) {}

void CLReduceMax::configure(ICLTensor *input, int axis, ICLTensor *output)
{
  _axis = axis;

  _input = input;
  _output = output;

  auto k = arm_compute::support::cpp14::make_unique<CLReduceMaxKernel>();
  k->configure(input, axis, output);
  _kernel = std::move(k);

  // We can handle for simple case only
  // Output rank: 1
  // Axis: one axis value, restrict to 1
  ARM_COMPUTE_ERROR_ON(input->info()->tensor_shape().num_dimensions() != 2);
  ARM_COMPUTE_ERROR_ON(output->info()->tensor_shape().num_dimensions() != 1);
  ARM_COMPUTE_ERROR_ON(axis != 1);
}

Status CLReduceMax::validate(const ITensorInfo *input, int32_t axis, const ITensorInfo *output)
{
  return CLReduceMaxKernel::validate(input, axis, output);
}

void CLReduceMax::run()
{
#if REDUCE_MAX_RUN_ON_CPU
  run_on_cpu();

  arm_compute::CLScheduler::get().sync();
#else
  arm_compute::CLScheduler::get().enqueue(*_kernel);
#endif
}

void CLReduceMax::run_on_cpu()
{
  cl::CommandQueue q = CLScheduler::get().queue();

  _input->map(q);
  _output->map(q);

  // Compute by CPU for simple case
  // Input rank: 2
  // Output rank: 1
  // Axis: one axis value, restrict to 1

  float *input_data = (float *)_input->buffer();
  float *output_data = (float *)_output->buffer();

  std::vector<float> container_max;
  int cols = _input->info()->tensor_shape()[0];
  int rows = _input->info()->tensor_shape()[1];
  container_max.resize(rows);

  // Initialize as 1st element in row
  float *input_pointer = input_data;
  for (int i = 0; i < rows; i++)
  {
    container_max[i] = *input_pointer;
    input_pointer += cols;
  }

  // Update max value in row
  for (int i = 0; i < rows; i++)
  {
    float max_in_row = container_max[i];
    for (int j = 1; j < cols; j++)
    {
      if (max_in_row < input_data[i * cols + j])
      {
        max_in_row = input_data[i * cols + j];
      }
    }
    container_max[i] = max_in_row;
  }

  for (int i = 0; i < rows; i++)
  {
    output_data[i] = container_max[i];
  }

  _input->unmap(q);
  _output->unmap(q);
}
} // namespace arm_compute
