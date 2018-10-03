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

// for 2D tensor
#ifndef __NNFW_UTIL_MATRIX_SHAPE_H__
#define __NNFW_UTIL_MATRIX_SHAPE_H__

#include <cstdint>

namespace nnfw
{
namespace util
{
namespace matrix
{

struct Shape
{
  int32_t H; // Height
  int32_t W; // Width

  Shape() = default;
  Shape(int32_t height, int32_t width) : H{height}, W{width}
  {
    // DO NOTHING
  }
};

} // namespace feature
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_MATRIX_SHAPE_H__
