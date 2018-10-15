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

#ifndef __NNFW_UTIL_KERNEL_INDEX_ITERATOR_H__
#define __NNFW_UTIL_KERNEL_INDEX_ITERATOR_H__

#include "util/kernel/Shape.h"

namespace nnfw
{
namespace util
{
namespace kernel
{

class IndexIterator
{
public:
  IndexIterator(const Shape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  template <typename Callable> IndexIterator &iter(Callable cb)
  {
    for (int32_t nth = 0; nth < _shape.N; ++nth)
    {
      for (int32_t ch = 0; ch < _shape.C; ++ch)
      {
        for (int32_t row = 0; row < _shape.H; ++row)
        {
          for (int32_t col = 0; col < _shape.W; ++col)
          {
            cb(nth, ch, row, col);
          }
        }
      }
    }

    return (*this);
  }

private:
  const Shape _shape;
};

inline IndexIterator iterate(const Shape &shape) { return IndexIterator{shape}; }

template <typename Callable> IndexIterator &operator<<(IndexIterator &&it, Callable cb)
{
  return it.iter(cb);
}

} // namespace kernel
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_INDEX_ITERATOR_H__
