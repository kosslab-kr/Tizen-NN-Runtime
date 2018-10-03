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
#include "arm_compute/core/CL/kernels/CLGatherKernel.h"

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

Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2,
                          const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::S32,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S32,
                                                       DataType::F32);

  return Status{};
}

} // namespace

CLGatherKernel::CLGatherKernel() : _input1(nullptr), _input2(nullptr), _output(nullptr) {}

void CLGatherKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, output);

  _input1 = input1;
  _input2 = input2;
  _output = output;

  // Construct kernel name
  std::string kernel_name = "gather";
  if (input1->info()->num_dimensions() == 1)
  {
    kernel_name = "gather_1d";
  }
  else if (input1->info()->num_dimensions() == 2)
  {
    if (_output->info()->num_dimensions() == 1)
    {
      kernel_name = "gather_1d_out";
    }
  }

  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(input1->info()->data_type()));
  build_opts.emplace("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(input2->info()->data_type()));
  build_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts));

  // Configure kernel window
  const unsigned int num_elems_processed_per_iteration = 1;
  Window win = calculate_max_window(*input2->info(), Steps(num_elems_processed_per_iteration));
  output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

  ICLKernel::configure(win);
}

Status CLGatherKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2,
                                const ITensorInfo *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output));

  return Status{};
}

void CLGatherKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  if (_input1->info()->num_dimensions() == 1)
  {
    Window slice = window.first_slice_window_1D();

    unsigned int idx = 0;
    add_1D_tensor_argument(idx, _input1, slice);
    add_1D_tensor_argument(idx, _input2, slice);
    add_1D_tensor_argument(idx, _output, slice);
    enqueue(queue, *this, slice);
  }
  else if (_input1->info()->num_dimensions() == 2)
  {
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimY);
    Window slice = window.collapse_if_possible(ICLKernel::window(), Window::DimX);

    // Set inputs
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _input1, window_collapsed);
    add_1D_tensor_argument(idx, _input2, slice);
    if (_output->info()->num_dimensions() == 1)
    {
      add_1D_tensor_argument(idx, _output, slice);
    }
    else
    {
      add_2D_tensor_argument(idx, _output, window_collapsed);
    }
    enqueue(queue, *this, slice);
  }
}
