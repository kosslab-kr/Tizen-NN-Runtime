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

#ifndef __NNFW_UTIL_TENSOR_OBJECT_H__
#define __NNFW_UTIL_TENSOR_OBJECT_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"
#include "util/tensor/IndexIterator.h"
#include "util/tensor/NonIncreasingStride.h"
#include "util/tensor/Reader.h"

#include <vector>

namespace nnfw
{
namespace util
{
namespace tensor
{

template <typename T> class Object final : public Reader<T>
{
public:
  using Generator = std::function<T(const Shape &shape, const Index &index)>;

public:
  Object(const Shape &shape, const Generator &fn) : _shape{shape}
  {
    // Set 'stride'
    _stride.init(shape);

    // Pre-allocate buffer
    _values.resize(_shape.dim(0) * _stride.at(0));

    // Set 'value'
    iterate(_shape) <<
        [this, &fn](const Index &index) { _values.at(_stride.offset(index)) = fn(_shape, index); };
  }

public:
  const Shape &shape(void) const { return _shape; }

public:
  T at(const Index &index) const override { return _values.at(_stride.offset(index)); }

private:
  Shape _shape;
  NonIncreasingStride _stride;

private:
  std::vector<T> _values;
};

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_OBJECT_H__
