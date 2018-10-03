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

#ifndef __NEURUN_LINEAR_LINEAR_H__
#define __NEURUN_LINEAR_LINEAR_H__

#include <vector>

#include "graph/operation/Node.h"
#include "backend/ITensorBuilder.h"

namespace neurun
{
namespace graph
{
namespace operation
{
struct NodeVisitor;
} // namespace operation
} // namespace graph
} // namespace neurun

namespace neurun
{
namespace graph
{
class Graph;
} // namespace graph
} // namespace neurun

namespace neurun
{
namespace linear
{

class Linear
{
public:
  Linear(const graph::Graph &graph);

public:
  Linear(const Linear &linear) = delete;

public:
  void accept(graph::operation::NodeVisitor &&visitor) const;

  // TODO Should not return TensorBuilderSet
  virtual backend::TensorBuilderSet markTensors() const;

public:
private:
  std::vector<const graph::operation::Node *> _operations;
};

} // namespace linear
} // namespace neurun

#endif // __NEURUN_LINEAR_LINEAR_H__
