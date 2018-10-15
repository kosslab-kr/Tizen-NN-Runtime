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

#ifndef __NNFW_UTIL_TENSOR_ZIPPER_H__
#define __NNFW_UTIL_TENSOR_ZIPPER_H__

#include "util/tensor/Index.h"
#include "util/tensor/IndexIterator.h"
#include "util/tensor/Reader.h"

namespace nnfw
{
namespace util
{
namespace tensor
{

template <typename T> class Zipper
{
public:
  Zipper(const Shape &shape, const Reader<T> &lhs, const Reader<T> &rhs)
      : _shape{shape}, _lhs{lhs}, _rhs{rhs}
  {
    // DO NOTHING
  }

public:
  template <typename Callable> void zip(Callable cb) const
  {
    iterate(_shape) <<
        [this, &cb](const Index &index) { cb(index, _lhs.at(index), _rhs.at(index)); };
  }

private:
  const Shape &_shape;
  const Reader<T> &_lhs;
  const Reader<T> &_rhs;
};

template <typename T, typename Callable>
const Zipper<T> &operator<<(const Zipper<T> &zipper, Callable cb)
{
  zipper.zip(cb);
  return zipper;
}

template <typename T> Zipper<T> zip(const Shape &shape, const Reader<T> &lhs, const Reader<T> &rhs)
{
  return Zipper<T>{shape, lhs, rhs};
}

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_ZIPPER_H__
