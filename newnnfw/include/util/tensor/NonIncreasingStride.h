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

#ifndef __NNFW_UTIL_TENSOR_NON_INCREASING_STRIDE_H__
#define __NNFW_UTIL_TENSOR_NON_INCREASING_STRIDE_H__

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"

#include <vector>

namespace nnfw
{
namespace util
{
namespace tensor
{

// As its name suggests, stride[N-1] >= stride[N] holds for all N < rank in NonIncreasingStride.
class NonIncreasingStride
{
public:
  void init(const Shape &shape)
  {
    _stride.resize(shape.rank());
    _stride.at(shape.rank() - 1) = 1;

    for (uint32_t axis = shape.rank() - 1; axis > 0; --axis)
    {
      _stride.at(axis - 1) = _stride.at(axis) * shape.dim(axis);
    }
  }

public:
  uint32_t at(uint32_t axis) const { return _stride.at(axis); }

public:
  uint32_t offset(const Index &index) const;

private:
  std::vector<uint32_t> _stride;
};

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_NON_INCREASING_STRIDE_H__
