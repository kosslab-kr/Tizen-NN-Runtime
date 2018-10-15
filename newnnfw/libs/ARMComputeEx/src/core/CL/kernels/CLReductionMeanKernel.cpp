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
#include "arm_compute/core/CL/kernels/CLReductionMeanKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          std::vector<uint32_t> axis)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis.size() >= TensorShape::num_max_dimensions,
                                  "Reduction axis greater than max number of dimensions");

  std::vector<uint32_t>::const_iterator it;
  bool axis_w = false;
  bool axis_h = false;
  for (it = axis.begin(); it != axis.end(); ++it)
  {
    if ((*it) == 0)
    {
      axis_w = true;
    }
    else if ((*it) == 1)
    {
      axis_h = true;
    }
    else
    {
      ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported axis!");
    }
  }
  // TODO Other axises (currently, only axises for both width and height are supported.)
  if (!axis_w || !axis_h)
  {
    ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported axis!");
  }

  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output->data_layout() != DataLayout::NCHW);
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output,
                                                         std::vector<uint32_t> axis)
{
  // Output tensor auto initialization if not yet initialized
  TensorShape output_shape{input->tensor_shape()};
  output_shape.set(0, 1);
  output_shape.set(1, 1);
  auto_init_if_empty(*output, output_shape, output->num_channels(), input->data_type(),
                     input->fixed_point_position());

  // Configure kernel window
  constexpr unsigned int num_elems_processed_per_iteration_x = 8; // step
  const unsigned int num_elems_processed_per_iteration_y = input->dimension(1);

  Window win = calculate_max_window(
      *input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
  AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x,
                                     num_elems_processed_per_iteration_y);
  AccessWindowHorizontal output_access(output, 0, 1);
  bool window_changed = update_window_and_padding(win, input_access, output_access);
  output_access.set_valid_region(win, output->valid_region());

  Status err = (window_changed)
                   ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!")
                   : Status{};

  return std::make_tuple(err, win);
}
} // namespace

CLReductionMeanKernel::CLReductionMeanKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(), _border_size()
{
}

BorderSize CLReductionMeanKernel::border_size() const { return _border_size; }

void CLReductionMeanKernel::configure(const ICLTensor *input, ICLTensor *output,
                                      std::vector<uint32_t> axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis));

  _input = input;
  _output = output;
  _reduction_axis = axis;

  constexpr unsigned int num_elems_processed_per_iteration_x = 8; // step

  // Set border size
  _border_size = BorderSize(
      ceil_to_multiple(input->info()->dimension(0), num_elems_processed_per_iteration_x) -
      input->info()->dimension(0));

  // Set build options
  std::set<std::string> build_opts;
  build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
  // build_opts.emplace(("-DVEC_SIZE=" +
  // support::cpp11::to_string(num_elems_processed_per_iteration)));
  if (is_data_type_fixed_point(input->info()->data_type()))
  {
    build_opts.emplace("-DFIXED_POINT_POSITION=" +
                       support::cpp11::to_string(input->info()->fixed_point_position()));
  }

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel("reduction_mean", build_opts));

  // Configure kernel window
  auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis);

  ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

  ICLKernel::configure(std::get<1>(win_config));
}

Status CLReductionMeanKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                       std::vector<uint32_t> axis)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis));
  ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(
      validate_and_configure_window(input->clone().get(), output->clone().get(), axis)));

  return Status{};
}

void CLReductionMeanKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  // Set out window
  Window out_window(window);
  out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

  // Get first input and output slices
  Window in_slice = window.first_slice_window_2D();
  Window out_slice = out_window.first_slice_window_2D();

  // Set local sums buffer
  // TODO work_group
  unsigned int local_sum_size = _lws_hint[0] * _input->info()->element_size();

  unsigned int idx = 2 * num_arguments_per_2D_tensor();
  _kernel.setArg(idx++, local_sum_size, nullptr);
  _kernel.setArg<cl_int>(idx++, static_cast<cl_int>(_input->info()->dimension(1))); // height
  _kernel.setArg<cl_int>(idx++, static_cast<cl_int>(_input->info()->dimension(0) *
                                                    _input->info()->dimension(1))); // divider

  do
  {
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _input, in_slice);
    in_slice.set_dimension_step(Window::DimY, _input->info()->dimension(1));
    add_2D_tensor_argument(idx, _output, out_slice);
    enqueue(queue, *this, in_slice);
  } while (window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
}
