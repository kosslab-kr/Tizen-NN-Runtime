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

#ifndef __NNFW_UTIL_FEATURE_OBJECT_H__
#define __NNFW_UTIL_FEATURE_OBJECT_H__

#include "util/feature/Shape.h"
#include "util/feature/Index.h"
#include "util/feature/Reader.h"

#include <vector>

namespace nnfw
{
namespace util
{
namespace feature
{

template <typename T> class Object final : public Reader<T>
{
public:
  using Generator = std::function<T(const Shape &shape, const Index &index)>;

public:
  Object(const Shape &shape, const Generator &fn) : _shape{shape}
  {
    _value.resize(_shape.C * _shape.H * _shape.W);

    for (int32_t ch = 0; ch < _shape.C; ++ch)
    {
      for (int32_t row = 0; row < _shape.H; ++row)
      {
        for (int32_t col = 0; col < _shape.W; ++col)
        {
          _value.at(offsetOf(ch, row, col)) = fn(_shape, Index{ch, row, col});
        }
      }
    }
  }

public:
  const Shape &shape(void) const { return _shape; }

public:
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _value.at(offsetOf(ch, row, col));
  }

private:
  uint32_t offsetOf(uint32_t ch, uint32_t row, uint32_t col) const
  {
    return ch * _shape.H * _shape.W + row * _shape.W + col;
  }

private:
  Shape _shape;
  std::vector<T> _value;
};

} // namespace feature
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_OBJECT_H__
