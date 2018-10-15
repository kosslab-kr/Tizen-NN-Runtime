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

#ifndef __NNFW_UTIL_FEATURE_SHAPE_H__
#define __NNFW_UTIL_FEATURE_SHAPE_H__

#include <cstdint>

namespace nnfw
{
namespace util
{
namespace feature
{

struct Shape
{
  int32_t N; // Batch
  int32_t C; // Depth
  int32_t H; // Height
  int32_t W; // Width

  Shape() = default;
  Shape(int32_t depth, int32_t height, int32_t width) : N{1}, C{depth}, H{height}, W{width}
  {
    // DO NOTHING
  }
  Shape(int32_t batch, int32_t depth, int32_t height, int32_t width) : N{batch}, C{depth}, H{height}, W{width}
  {
    // DO NOTHING
  }
};

} // namespace feature
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_H__
