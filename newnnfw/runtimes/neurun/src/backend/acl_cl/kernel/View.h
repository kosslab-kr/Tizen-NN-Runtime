/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __INTERNAL_ARM_COMPUTE_KERNEL_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_KERNEL_VIEW_H__

#include "util/kernel/Shape.h"
#include "util/kernel/Reader.h"

#include <arm_compute/core/ITensor.h>

#include <cassert>

namespace internal
{
namespace arm_compute
{
namespace kernel
{

template <typename T> class View;

template <> class View<float> final : public nnfw::util::kernel::Reader<float>
{
public:
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    assert(tensor->info()->data_type() == ::arm_compute::DataType::F32);

    _shape.N = tensor->info()->dimension(3);
    _shape.C = tensor->info()->dimension(2);
    _shape.H = tensor->info()->dimension(1);
    _shape.W = tensor->info()->dimension(0);
  }

public:
  const ::nnfw::util::kernel::Shape &shape(void) const { return _shape; }

public:
  float at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    const auto offset = kernel_index_to_byte_offset(nth, ch, row, col);

    float *ptr = reinterpret_cast<float *>(_tensor->buffer() + offset);

    return *ptr;
  }

public:
  float &at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col)
  {
    const auto offset = kernel_index_to_byte_offset(nth, ch, row, col);

    float *ptr = reinterpret_cast<float *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  size_t kernel_index_to_byte_offset(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const
  {
    return _tensor->info()->offset_element_in_bytes(::arm_compute::Coordinates{col, row, ch, nth});
  }

private:
  ::nnfw::util::kernel::Shape _shape;
  ::arm_compute::ITensor *_tensor;
};

} // namespace kernel
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_FEATURE_VIEW_H__
