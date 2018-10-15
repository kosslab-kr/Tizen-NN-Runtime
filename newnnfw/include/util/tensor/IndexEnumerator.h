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

#ifndef __NNFW_UTIL_TENSOR_INDEX_ENUMERATOR_H__
#define __NNFW_UTIL_TENSOR_INDEX_ENUMERATOR_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"

namespace nnfw
{
namespace util
{
namespace tensor
{

class IndexEnumerator
{
public:
  explicit IndexEnumerator(const Shape &shape) : _shape(shape), _index(shape.rank()), _cursor(0)
  {
    const size_t rank = _shape.rank();

    for (size_t axis = 0; axis < rank; ++axis)
    {
      _index.at(axis) = 0;
    }

    for (_cursor = 0; _cursor < rank; ++_cursor)
    {
      if (_index.at(_cursor) < _shape.dim(_cursor))
      {
        break;
      }
    }
  }

public:
  IndexEnumerator(IndexEnumerator &&) = delete;
  IndexEnumerator(const IndexEnumerator &) = delete;

public:
  bool valid(void) const { return _cursor < _shape.rank(); }

public:
  const Index &curr(void) const { return _index; }

public:
  void advance(void)
  {
    const size_t rank = _shape.rank();

    // Find axis to be updated
    while((_cursor < rank) && !(_index.at(_cursor) + 1 < _shape.dim(_cursor)))
    {
      ++_cursor;
    }

    if(_cursor == rank)
    {
      return;
    }

    // Update index
    _index.at(_cursor) += 1;

    for (size_t axis = 0; axis < _cursor; ++axis)
    {
      _index.at(axis) = 0;
    }

    // Update cursor
    _cursor = 0;
  }

public:
  const Shape _shape;

private:
  size_t _cursor;
  Index _index;
};

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_INDEX_ENUMERATOR_H__
