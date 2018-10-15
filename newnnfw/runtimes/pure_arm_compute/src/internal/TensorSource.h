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

#ifndef __INTERNAL_TENSOR_SOURCE_H__
#define __INTERNAL_TENSOR_SOURCE_H__

#include <util/tensor/Shape.h>
#include <util/tensor/IndexIterator.h>

#include "internal/Source.h"
#include "internal/Swizzle.h"
#include "internal/nnapi/tensor/Reader.h"
#include "internal/arm_compute/tensor/View.h"

// NOTE TensorSource is much slower than specialized Source(s)
template <typename T> class TensorSource final : public Source
{
public:
  TensorSource(const nnfw::util::tensor::Shape &shape, const T *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  void push(::arm_compute::ITensor &tensor) const override
  {
    const ::internal::nnapi::tensor::Reader<T> from{_shape, _base, _size};
    ::internal::arm_compute::tensor::View<T> into{&tensor};

    ::nnfw::util::tensor::iterate(_shape) << [&](const nnfw::util::tensor::Index &index_nnapi) {
      const auto rank = index_nnapi.rank();
      nnfw::util::tensor::Index index_ACL(rank);

      for (uint32_t axis = 0; axis < rank; ++axis)
      {
        index_ACL.at(ToARMComputeAxis(rank, axis).value()) = index_nnapi.at(axis);
      }

      into.at(index_ACL) = from.at(index_nnapi);
    };
  }

private:
  const nnfw::util::tensor::Shape _shape;
  const T *const _base;
  const size_t _size;
};

#endif // __INTERNAL_TENSOR_SOURCE_H__
