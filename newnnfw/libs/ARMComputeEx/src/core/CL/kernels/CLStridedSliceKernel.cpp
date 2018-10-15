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
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <string>

using namespace std;
using namespace arm_compute;

static const int32_t maxDim = 4;

CLStridedSliceKernel::CLStridedSliceKernel()
    : _input(nullptr), _output(nullptr), _beginData(nullptr), _endData(nullptr),
      _stridesData(nullptr), _beginMask(0), _endMask(0), _shrinkAxisMask(0)
{
}

Status CLStridedSliceKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                      const ITensorInfo *begin, const ITensorInfo *end,
                                      const ITensorInfo *strides, int32_t beginMask,
                                      int32_t endMask, int32_t shrinkAxisMask)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, begin, end, strides);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
      input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::QASYMM8, DataType::U16,
      DataType::S16, DataType::QS16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(begin, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(end, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(strides, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

  ARM_COMPUTE_ERROR_ON(begin->num_dimensions() != 1 || begin->dimension(0) > 4);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(begin->tensor_shape(), end->tensor_shape(),
                                              strides->tensor_shape());

  return Status{};
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axisSize - 1] that can be used to index
// directly into the data.
inline int32_t StartForAxis(int32_t beginMask, int32_t begin, int32_t stride,
                            const TensorShape &inputShape, int32_t axis)
{
  // Begin with the specified index
  int32_t start = begin;

  // beginMask override
  if (beginMask & 1 << axis)
  {
    if (stride > 0)
    {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axisSize-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = std::numeric_limits<int32_t>::lowest();
    }
    else
    {
      // Backward iteration - use the last element.
      start = std::numeric_limits<int32_t>::max();
    }
  }

  // Handle negative indices
  int32_t axisSize = inputShape[axis];
  if (start < 0)
  {
    start += axisSize;
  }

  // Clamping
  start = arm_compute::utility::clamp(start, 0, axisSize - 1);

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
inline int32_t StopForAxis(int32_t endMask, int32_t end, int32_t stride,
                           const TensorShape &inputShape, int32_t axis)
{
  // Begin with the specified index
  int32_t stop = end;

  // endMask override
  if (endMask & (1 << axis))
  {
    if (stride > 0)
    {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = std::numeric_limits<int32_t>::max();
    }
    else
    {
      // Backward iteration - use the first element.
      stop = std::numeric_limits<int32_t>::lowest();
    }
  }

  // Handle negative indices
  int32_t axisSize = inputShape[axis];
  if (stop < 0)
  {
    stop += axisSize;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (stride > 0)
  {
    // Forward iteration
    stop = arm_compute::utility::clamp(stop, 0, axisSize);
  }
  else
  {
    // Backward iteration
    stop = arm_compute::utility::clamp(stop, -1, axisSize - 1);
  }

  return stop;
}

inline int32_t offset4D(const TensorShape &shape, int32_t b, int32_t d, int32_t h, int32_t w)
{
  int32_t offset = b * shape[2] * shape[1] * shape[0];
  offset += d * shape[1] * shape[0];
  offset += h * shape[0];
  offset += w;
  return offset;
}

inline int32_t getOutDim(int32_t start, int32_t stop, int32_t stride)
{
  int32_t ret = 0;
  if (stride > 0)
  {
    ret = ((stop - start - 1) / stride) + 1;
  }
  else
  {
    ret = ((stop - start + 1) / stride) + 1;
  }
  ARM_COMPUTE_ERROR_ON_MSG(ret < 0, "The dimension must be the natural number");
  return ret;
}

void CLStridedSliceKernel::configure(const ICLTensor *input, ICLTensor *output,
                                     ICLTensor *beginData, ICLTensor *endData,
                                     ICLTensor *stridesData, int32_t beginMask, int32_t endMask,
                                     int32_t shrinkAxisMask)
{
  ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), beginData->info(),
                                      endData->info(), stridesData->info(), beginMask, endMask,
                                      shrinkAxisMask));

  _input = input;
  _output = output;
  _beginData = beginData;
  _endData = endData;
  _stridesData = stridesData;
  _beginMask = beginMask;
  _endMask = endMask;
  _shrinkAxisMask = shrinkAxisMask;

  constexpr unsigned int num_elems_processed_per_iteration = 1;

  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-DELEMENT_DATA_TYPE=" +
                     get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.emplace("-DELEMENT_SIZE=" + support::cpp11::to_string(input->info()->element_size()));

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel("strided_slice", build_opts));

  // Create output's window without padding
  TensorShape collapsed = output->info()->tensor_shape();
  collapsed.collapse(4);
  TensorInfo info = *output->info();
  info.set_tensor_shape(collapsed);
  Window win = calculate_max_window(info, Steps(num_elems_processed_per_iteration));

  ICLKernel::configure(win);
}

void CLStridedSliceKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  // Create input window
  TensorShape collapsed = _input->info()->tensor_shape();
  collapsed.collapse(4);
  TensorInfo info = *_input->info();
  info.set_tensor_shape(collapsed);
  Window win_in = calculate_max_window(info, Steps(_input->info()->tensor_shape().total_size()));

  _beginData->map(queue);
  _endData->map(queue);
  _stridesData->map(queue);

  std::vector<int32_t> dimsIn;
  std::vector<int32_t> dimsOut;
  std::vector<int32_t> starts;
  std::vector<int32_t> stops;
  std::vector<int32_t> strides;

  for (uint32_t n = 0; n < _beginData->info()->tensor_shape().total_size(); ++n)
  {
    const TensorShape shape = _input->info()->tensor_shape();
    starts.emplace_back(
        StartForAxis(_beginMask, reinterpret_cast<int32_t *>(_beginData->buffer())[n],
                     reinterpret_cast<int32_t *>(_stridesData->buffer())[n], shape, n));

    stops.emplace_back(StopForAxis(_endMask, reinterpret_cast<int32_t *>(_endData->buffer())[n],
                                   reinterpret_cast<int32_t *>(_stridesData->buffer())[n], shape,
                                   n));

    strides.emplace_back(reinterpret_cast<int32_t *>(_stridesData->buffer())[n]);
    dimsIn.emplace_back(shape[n]);
    dimsOut.emplace_back(getOutDim(starts[n], stops[n], strides[n]));
  }

  for (uint32_t n = _beginData->info()->tensor_shape().total_size(); n < 4; n++)
  {
    starts.emplace_back(0);
    stops.emplace_back(1);
    strides.emplace_back(1);
    dimsIn.emplace_back(1);
    dimsOut.emplace_back(1);
  }
  // TODO: Apply shrinkAxisMask

  _beginData->unmap(queue);
  _stridesData->unmap(queue);
  _endData->unmap(queue);

  // Set parameters
  unsigned int idx = 2 * num_arguments_per_1D_tensor(); // Skip the input and output parameters
  const cl_int4 dimsInArg = {{
      static_cast<cl_int>(dimsIn[0]), static_cast<cl_int>(dimsIn[1]),
      static_cast<cl_int>(dimsIn[2]), static_cast<cl_int>(dimsIn[3]),
  }};
  _kernel.setArg<cl_int4>(idx++, dimsInArg);

  const cl_int4 dimsOutArg = {{
      static_cast<cl_int>(dimsOut[0]), static_cast<cl_int>(dimsOut[1]),
      static_cast<cl_int>(dimsOut[2]), static_cast<cl_int>(dimsOut[3]),
  }};
  _kernel.setArg<cl_int4>(idx++, dimsOutArg);

  const cl_int4 startsArg = {{
      static_cast<cl_int>(starts[0]), static_cast<cl_int>(starts[1]),
      static_cast<cl_int>(starts[2]), static_cast<cl_int>(starts[3]),
  }};
  _kernel.setArg<cl_int4>(idx++, startsArg);

  const cl_int4 stridesArg = {{
      static_cast<cl_int>(strides[0]), static_cast<cl_int>(strides[1]),
      static_cast<cl_int>(strides[2]), static_cast<cl_int>(strides[3]),
  }};
  _kernel.setArg<cl_int4>(idx++, stridesArg);

  // TODO: Apply slicing output's window
  idx = 0;
  add_1D_tensor_argument(idx, _input, win_in);
  add_1D_tensor_argument(idx, _output, window);

  enqueue(queue, *this, window);
}
