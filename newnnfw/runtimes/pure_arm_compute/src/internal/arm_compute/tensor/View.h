/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{
namespace tensor
{

template <typename T> class View
{
public:
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

private:
  uint32_t byte_offset_of(const nnfw::util::tensor::Index &index) const
  {
    // NOTE index.rank() >= _tensor->info()->num_dimensions() should hold here
    const uint32_t rank = index.rank();

    ::arm_compute::Coordinates coordinates;

    coordinates.set_num_dimensions(rank);

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      coordinates[axis] = index.at(axis);
    }

    return _tensor->info()->offset_element_in_bytes(coordinates);
  }

public:
  T at(const nnfw::util::tensor::Index &index) const
  {
    const auto offset = byte_offset_of(index);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

  T &at(const nnfw::util::tensor::Index &index)
  {
    const auto offset = byte_offset_of(index);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  ::arm_compute::ITensor *_tensor;
};

} // namespace tensor
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__
