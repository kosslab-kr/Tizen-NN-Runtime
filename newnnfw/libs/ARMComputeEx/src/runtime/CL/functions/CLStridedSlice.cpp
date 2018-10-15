/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2017 ARM Limited.
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#include "arm_compute/runtime/CL/functions/CLStridedSlice.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"
#include <vector>

using namespace arm_compute;

static const int32_t maxDims = 4;

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axisSize - 1] that can be used to index
// directly into the data.
inline int32_t StartForAxis(int32_t beginMask, std::vector<int32_t> const &startIndices,
                            std::vector<int32_t> const &strides, const TensorShape &inputShape,
                            int32_t axis)
{
  // Begin with the specified index
  int32_t start = startIndices[axis];

  // beginMask override
  if (beginMask & 1 << axis)
  {
    if (strides[axis] > 0)
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
inline int32_t StopForAxis(int32_t endMask, std::vector<int32_t> const &stopIndices,
                           std::vector<int32_t> const &strides, const TensorShape &inputShape,
                           int32_t axis)
{
  // Begin with the specified index
  int32_t stop = stopIndices[axis];

  // endMask override
  if (endMask & (1 << axis))
  {
    if (strides[axis] > 0)
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
  if (strides[axis] > 0)
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

void CLStridedSlice::configure(const ICLTensor *input, ICLTensor *output, ICLTensor *beginData,
                               ICLTensor *endData, ICLTensor *stridesData, int32_t beginMask,
                               int32_t endMask, int32_t shrinkAxisMask)
{
  auto k = arm_compute::support::cpp14::make_unique<CLStridedSliceKernel>();
  k->configure(input, output, beginData, endData, stridesData, beginMask, endMask, shrinkAxisMask);
  _kernel = std::move(k);
}

void CLStridedSliceCPU::configure(ICLTensor *input, ICLTensor *output, ICLTensor *beginData,
                                  ICLTensor *endData, ICLTensor *stridesData, int32_t beginMask,
                                  int32_t endMask, int32_t shrinkAxisMask)
{
  ARM_COMPUTE_ERROR_THROW_ON(CLStridedSliceKernel::validate(
      input->info(), output->info(), beginData->info(), endData->info(), stridesData->info(),
      beginMask, endMask, shrinkAxisMask));

  _input = input;
  _output = output;
  _beginData = beginData;
  _endData = endData;
  _stridesData = stridesData;
  _beginMask = beginMask;
  _endMask = endMask;
  _shrinkAxisMask = shrinkAxisMask;
}

void CLStridedSliceCPU::run()
{
  run_on_cpu();

  arm_compute::CLScheduler::get().sync();
}

inline int32_t getOutDim(int32_t start, int32_t stop, int32_t stride)
{
  if (stride > 0)
  {
    return ((stop - start - 1) / stride) + 1;
  }
  else
  {
    return ((stop - start + 1) / stride) + 1;
  }
}

template <typename T>
inline void StridedSlice(const T *inputData, const TensorShape &inputShape, int32_t beginMask,
                         int32_t endMask, const std::vector<int32_t> &startIndices,
                         const std::vector<int32_t> &stopIndices,
                         const std::vector<int32_t> &strides, T *outputData)
{
  ARM_COMPUTE_ERROR_ON(startIndices.size() != maxDims);
  ARM_COMPUTE_ERROR_ON(stopIndices.size() != maxDims);
  ARM_COMPUTE_ERROR_ON(strides.size() != maxDims);

  const int32_t start_b = StartForAxis(beginMask, startIndices, strides, inputShape, 3);
  const int32_t stop_b = StopForAxis(endMask, stopIndices, strides, inputShape, 3);
  const int32_t start_d = StartForAxis(beginMask, startIndices, strides, inputShape, 2);
  const int32_t stop_d = StopForAxis(endMask, stopIndices, strides, inputShape, 2);
  const int32_t start_h = StartForAxis(beginMask, startIndices, strides, inputShape, 1);
  const int32_t stop_h = StopForAxis(endMask, stopIndices, strides, inputShape, 1);
  const int32_t start_w = StartForAxis(beginMask, startIndices, strides, inputShape, 0);
  const int32_t stop_w = StopForAxis(endMask, stopIndices, strides, inputShape, 0);

  // The shape of outputData may collapse in one-dimension.
  // Therefore, it is necessary to create a shape that matches the result of the outputData.
  TensorShape outputShape(
      getOutDim(start_w, stop_w, strides[0]), getOutDim(start_h, stop_h, strides[1]),
      getOutDim(start_d, stop_d, strides[2]), getOutDim(start_b, stop_b, strides[3]));
  for (int32_t in_b = start_b, b = 0; strides[3] > 0 ? in_b < stop_b : in_b > stop_b;
       in_b += strides[3], b++)
  {
    for (int32_t in_d = start_d, d = 0; strides[2] > 0 ? in_d < stop_d : in_d > stop_d;
         in_d += strides[2], d++)
    {
      for (int32_t in_h = start_h, h = 0; strides[1] > 0 ? in_h < stop_h : in_h > stop_h;
           in_h += strides[1], h++)
      {
        for (int32_t in_w = start_w, w = 0; strides[0] > 0 ? in_w < stop_w : in_w > stop_w;
             in_w += strides[0], w++)
        {
          outputData[offset4D(outputShape, b, d, h, w)] =
              inputData[offset4D(inputShape, in_b, in_d, in_h, in_w)];
        }
      }
    }
  }
}

void CLStridedSliceCPU::run_on_cpu()
{
  // TODO: Support shrinkAxisMask
  cl::CommandQueue q = CLScheduler::get().queue();

  _input->map(q);
  _output->map(q);
  _beginData->map(q);
  _endData->map(q);
  _stridesData->map(q);

  TensorShape inputShape = _input->info()->tensor_shape();
  TensorShape outputShape = _output->info()->tensor_shape();

  std::vector<int32_t> starts;
  std::vector<int32_t> stops;
  std::vector<int32_t> strides;

  for (uint32_t idx = 0; idx <= _input->info()->num_dimensions() - 1; ++idx)
  {
    starts.emplace_back(reinterpret_cast<int32_t *>(_beginData->buffer())[idx]);
    stops.emplace_back(reinterpret_cast<int32_t *>(_endData->buffer())[idx]);
    strides.emplace_back(reinterpret_cast<int32_t *>(_stridesData->buffer())[idx]);
  }

  for (uint32_t i = _input->info()->num_dimensions(); i < maxDims; i++)
  {
    starts.emplace_back(0);
    stops.emplace_back(1);
    strides.emplace_back(1);
  }

  switch (_input->info()->data_type())
  {
    case DataType::U8:
    case DataType::QASYMM8:
      StridedSlice(reinterpret_cast<const uint8_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides,
                   reinterpret_cast<uint8_t *>(_output->buffer()));
      break;
    case DataType::S8:
    case DataType::QS8:
      StridedSlice(reinterpret_cast<const int8_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides, reinterpret_cast<int8_t *>(_output->buffer()));
      break;
    case DataType::U16:
      StridedSlice(reinterpret_cast<const uint16_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides,
                   reinterpret_cast<uint16_t *>(_output->buffer()));
      break;
    case DataType::S16:
    case DataType::QS16:
      StridedSlice(reinterpret_cast<const int16_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides,
                   reinterpret_cast<int16_t *>(_output->buffer()));
      break;
    case DataType::F16:
      // Not sure this works.
      StridedSlice(reinterpret_cast<const half *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides, reinterpret_cast<half *>(_output->buffer()));
      break;
    case DataType::U32:
      StridedSlice(reinterpret_cast<const uint32_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides,
                   reinterpret_cast<uint32_t *>(_output->buffer()));
      break;
    case DataType::S32:
      StridedSlice(reinterpret_cast<const int32_t *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides,
                   reinterpret_cast<int32_t *>(_output->buffer()));
      break;
    case DataType::F32:
      StridedSlice(reinterpret_cast<const float *>(_input->buffer()), inputShape, _beginMask,
                   _endMask, starts, stops, strides, reinterpret_cast<float *>(_output->buffer()));
      break;
    default:
      ARM_COMPUTE_ERROR("DataType not supported");
      break;
  }

  _input->unmap(q);
  _output->unmap(q);
  _beginData->unmap(q);
  _endData->unmap(q);
  _stridesData->unmap(q);
}
