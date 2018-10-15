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

#ifndef __INTERNAL_NNAPI_TENSOR_VIEW_H__
#define __INTERNAL_NNAPI_TENSOR_VIEW_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"

namespace internal
{
namespace nnapi
{
namespace tensor
{

template <typename T> class View
{
public:
  // NOTE The parameter len denotes the number of bytes.
  View(const ::nnfw::util::tensor::Shape &shape, T *ptr, size_t len) : _shape{shape}, _ptr{ptr}
  {
    assert(shape.element_nums() * sizeof(T) == len);
  }

public:
  const nnfw::util::tensor::Shape &shape(void) const { return _shape; }

private:
  uint32_t offset_of(const nnfw::util::tensor::Index &index) const
  {
    if (_shape.rank() == 0)
    {
      return 0;
    }

    uint32_t offset = index.at(0);

    // Stride decreases as axis increases in NNAPI
    for (uint32_t axis = 1; axis < _shape.rank(); ++axis)
    {
      offset *= _shape.dim(axis);
      offset += index.at(axis);
    }

    return offset;
  }

public:
  T at(const nnfw::util::tensor::Index &index) const
  {
    const auto offset = offset_of(index);

    return _ptr[offset];
  }

  T &at(const nnfw::util::tensor::Index &index)
  {
    const auto offset = offset_of(index);

    return _ptr[offset];
  }

private:
  nnfw::util::tensor::Shape _shape;

private:
  T *_ptr;
};

} // namespace tensor
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_TENSOR_VIEW_H__
