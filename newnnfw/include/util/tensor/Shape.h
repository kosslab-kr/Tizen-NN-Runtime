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

#ifndef __NNFW_UTIL_TENSOR_SHAPE_H__
#define __NNFW_UTIL_TENSOR_SHAPE_H__

#include <cstdint>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <ostream>
#include <string>

namespace nnfw
{
namespace util
{
namespace tensor
{

class Shape
{
public:
  Shape(size_t rank) { _dimensions.resize(rank); }

public:
  Shape(const std::initializer_list<int32_t> &dimensions) : _dimensions{dimensions}
  {
    // DO NOTHING
  }

  Shape(const Shape &origin) = default;

public:
  void prepend(int32_t d) { _dimensions.emplace_front(d); }
  void append(int32_t d) { _dimensions.emplace_back(d); }

public:
  size_t rank(void) const { return _dimensions.size(); }

public:
  int32_t dim(size_t n) const { return _dimensions.at(n); }
  int32_t &dim(size_t n) { return _dimensions.at(n); }

public:
  size_t element_nums() const
  {
    size_t nums = 1;
    for (auto d : _dimensions)
    {
      nums *= d;
    }
    return nums;
  }

private:
  std::deque<int32_t> _dimensions;

public:
  static Shape from(const std::string &s);
};

bool operator==(const Shape &, const Shape &);

std::ostream &operator<<(std::ostream &os, const Shape &shape);

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_SHAPE_H__
