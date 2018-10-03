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

#ifndef __NNFW_SUPPORT_TFLITE_TENSOR_SHAPE_UTILS_H__
#define __NNFW_SUPPORT_TFLITE_TENSOR_SHAPE_UTILS_H__

#include "util/tensor/Shape.h"

#include <vector>

namespace nnfw
{
namespace support
{
namespace tflite
{

// Converts tensor::Shape into a vector
static inline std::vector<int32_t> as_dims(const nnfw::util::tensor::Shape &shape)
{
  std::vector<int32_t> dims;

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    dims.emplace_back(shape.dim(axis));
  }

  return dims;
}

nnfw::util::tensor::Shape broadcast(const nnfw::util::tensor::Shape &lhs_shape,
                                    const nnfw::util::tensor::Shape &rhs_shape);

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_TENSOR_SHAPE_UTILS_H__
