/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>

#include "Shape.h"

namespace neurun
{
namespace graph
{
namespace operand
{

Shape::Shape(uint32_t rank) { _dims.resize(rank); }

int32_t Shape::asVector(void) const
{
  assert(rank() == 1);

  return dim(0);
}

nnfw::util::feature::Shape Shape::asFeature(void) const
{
  assert(rank() == 4);

  // Feature Map in NNAPI
  //  - Dimension(0) -> Batch
  //  - Dimension(1) -> Height
  //  - Dimension(2) -> Width
  //  - Dimension(3) -> Depth
  const auto batch = dim(0);
  const auto depth = dim(3);
  const auto height = dim(1);
  const auto width = dim(2);

  return nnfw::util::feature::Shape(batch, depth, height, width);
}

nnfw::util::kernel::Shape Shape::asKernel(void) const
{
  assert(rank() == 4);

  // Convolution Kernel in NNAPI
  //  - Dimension(0) -> Count
  //  - Dimension(1) -> Height
  //  - Dimension(2) -> Width
  //  - Dimension(3) -> Depth
  const auto count = dim(0);
  const auto depth = dim(3);
  const auto height = dim(1);
  const auto width = dim(2);

  return nnfw::util::kernel::Shape(count, depth, height, width);
}

} // namespace operand
} // namespace graph
} // namespace neurun
