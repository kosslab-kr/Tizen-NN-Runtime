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

#ifndef __NNFW_UTIL_KERNEL_RANDOM_OBJECT_H__
#define __NNFW_UTIL_KERNEL_RANDOM_OBJECT_H__

#include "util/kernel/Shape.h"
#include "util/kernel/Reader.h"

#include <vector>

namespace nnfw
{
namespace util
{
namespace kernel
{

template <typename T> class RandomObject final : public Reader<T>
{
public:
  RandomObject(const Shape &shape) : _shape{shape}
  {
    const uint32_t size = _shape.N * _shape.C * _shape.H * _shape.W;

    // TODO Use random number
    for (uint32_t off = 0; off < size; ++off)
    {
      _value.emplace_back(static_cast<float>(off));
    }
  }

public:
  const Shape &shape(void) const { return _shape; }

public:
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = 0;

    index += nth * _shape.C * _shape.H * _shape.W;
    index += ch * _shape.H * _shape.W;
    index += row * _shape.W;
    index += col;

    return _value.at(index);
  }

private:
  const Shape _shape;
  std::vector<T> _value;
};

} // namespace kernel
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_KERNEL_RANDOM_OBJECT_H__
