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
#include "arm_compute/core/CL/kernels/CLReduceMaxKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>
#include <cstdlib>
#include <set>
#include <string>

using namespace arm_compute;

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_arguments(const ITensorInfo *input, int32_t axis, const ITensorInfo *output)
{
  // We can handle for simple case only
  // Input rank: 2
  // Output rank: 1
  // Axis: one axis value, restrict to 1

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis != 1, "Axis only allowed 1");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->tensor_shape().total_size() == 0,
                                  "Inputs are not broadcast compatible");

  // Validate in case of configured output
  if (output->total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() != input->data_type(),
                                    "Output same type allowed for input and output");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->tensor_shape().num_dimensions() != 1,
                                    "Only support for output dimension 1");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->tensor_shape().num_dimensions() != 2,
                                    "Only support for input dimension 2");
  }

  return Status{};
}

} // namespace

CLReduceMaxKernel::CLReduceMaxKernel() : _input(nullptr), _output(nullptr), _axis(0) {}

void CLReduceMaxKernel::configure(const ICLTensor *input, int32_t axis, ICLTensor *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), axis, output->info()));

  _input = input;
  _output = output;
  _axis = axis;

  // Configure kernel window
  int cols = _input->info()->tensor_shape()[0];
  int rows = _input->info()->tensor_shape()[1];
  Window win;
  win.set(0, Window::Dimension(0, cols, 1));
  win.set(1, Window::Dimension(0, rows, 1));

  // Construct kernel name
  std::string kernel_name = "reduce_max";

  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-DWIDTH=" + support::cpp11::to_string(cols));

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts));

  ICLKernel::configure(win);
}

Status CLReduceMaxKernel::validate(const ITensorInfo *input, int32_t axis,
                                   const ITensorInfo *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, axis, output));

  return Status{};
}

void CLReduceMaxKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

  Window window_input = window;
  Window slice_input = window_input.first_slice_window_1D();

  do
  {
    Window slice_output = slice_input.shift_dimensions(1);
    unsigned int idx = 0;
    add_1D_tensor_argument(idx, _input, slice_input);
    add_1D_tensor_argument(idx, _output, slice_output);
    enqueue(queue, *this, slice_input);

  } while (window_input.slide_window_slice_1D(slice_input));
}
