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

#ifndef __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__

#include "util/matrix/Shape.h"
#include "util/matrix/Reader.h"

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{
namespace matrix
{

template <typename T> class View final : public nnfw::util::matrix::Reader<T>
{
public:
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

public:
  T at(uint32_t row, uint32_t col) const override
  {
    const auto offset = matrix_index_to_byte_offset(row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

public:
  T &at(uint32_t row, uint32_t col)
  {
    const auto offset = matrix_index_to_byte_offset(row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  size_t matrix_index_to_byte_offset(uint32_t row, uint32_t col) const
  {
    return _tensor->info()->offset_element_in_bytes(::arm_compute::Coordinates{col, row});
  }

private:
  ::arm_compute::ITensor *_tensor;
};

} // namespace matrix
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__
