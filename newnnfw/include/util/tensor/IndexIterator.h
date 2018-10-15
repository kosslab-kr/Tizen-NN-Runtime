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

#ifndef __NNFW_UTIL_TENSOR_INDEX_ITERATOR_H__
#define __NNFW_UTIL_TENSOR_INDEX_ITERATOR_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"
#include "util/tensor/IndexEnumerator.h"

namespace nnfw
{
namespace util
{
namespace tensor
{

class IndexIterator
{
public:
  IndexIterator(const Shape &shape) : _shape(shape)
  {
    // DO NOTHING
  }

public:
  // Allow move, but disallow copy
  IndexIterator(IndexIterator &&) = default;
  IndexIterator(const IndexIterator &) = delete;

public:
  template <typename Callable> IndexIterator &iter(Callable fn)
  {
    for (IndexEnumerator e{_shape}; e.valid(); e.advance())
    {
      fn(e.curr());
    }

    return (*this);
  }

private:
  const Shape &_shape;
};

inline IndexIterator iterate(const Shape &shape) { return IndexIterator{shape}; }

template <typename Callable> IndexIterator &operator<<(IndexIterator &&it, Callable cb)
{
  return it.iter(cb);
}

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_INDEX_ITERATOR_H__
