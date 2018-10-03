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

#ifndef __NNFW_UTIL_VECTOR_OBJECT_H__
#define __NNFW_UTIL_VECTOR_OBJECT_H__

#include "util/vector/Reader.h"

#include <vector>
#include <functional>

namespace nnfw
{
namespace util
{
namespace vector
{

template <typename T> class Object final : public Reader<T>
{
public:
  using Generator = std::function<T(int32_t size, int32_t offset)>;

public:
  Object(int32_t size, const Generator &gen) : _size{size}
  {
    _value.resize(_size);

    for (int32_t offset = 0; offset < size; ++offset)
    {
      _value.at(offset) = gen(size, offset);
    }
  }

public:
  int32_t size(void) const { return _size; }

public:
  T at(uint32_t nth) const override { return _value.at(nth); }

private:
  const int32_t _size;
  std::vector<T> _value;
};

} // namespace vector
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_VECTOR_OBJECT_H__
