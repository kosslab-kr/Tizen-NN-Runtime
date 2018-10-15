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

#ifndef __NEURUN_GRAPH_OPERAND_SHAPE_H__
#define __NEURUN_GRAPH_OPERAND_SHAPE_H__

#include <vector>
#include <cstdint>

#include "util/feature/Shape.h"
#include "util/kernel/Shape.h"

namespace neurun
{
namespace graph
{
namespace operand
{

struct Shape
{
public:
  Shape(uint32_t rank);

public:
  uint32_t rank(void) const { return _dims.size(); }

public:
  int32_t dim(uint32_t n) const { return _dims.at(n); }
  int32_t &dim(uint32_t n) { return _dims.at(n); }
  const std::vector<int32_t> &dims() const { return _dims; }

public:
  int32_t asVector(void) const;
  nnfw::util::feature::Shape asFeature(void) const;
  nnfw::util::kernel::Shape asKernel(void) const;

private:
  std::vector<int32_t> _dims;
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_SHAPE_H__
