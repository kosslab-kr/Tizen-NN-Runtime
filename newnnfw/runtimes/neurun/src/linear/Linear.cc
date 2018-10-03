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

#include "Linear.h"

#include "graph/Graph.h"

#include "graph/operation/LowerInfo.h"
#include "backend/IStageGenerator.h"

namespace neurun
{
namespace linear
{

Linear::Linear(const graph::Graph &graph)
{
  // Linearize with topological sort
  //
  // Topological sort algorithm
  //   1. Iterate with DFS
  //   2. Append the node to vector when DFS for the node finishes(post order)
  //   3. Reverse the order of nodes

  graph::Graph::PostDfsConstIterator().iterate(
      graph, [&](const neurun::graph::operation::Node &node) { _operations.emplace_back(&node); });

  std::reverse(std::begin(_operations), std::end(_operations));
}

void Linear::accept(graph::operation::NodeVisitor &&visitor) const
{
  for (const auto op : _operations)
  {
    op->accept(std::move(visitor));
  }
}

backend::TensorBuilderSet Linear::markTensors() const
{
  backend::TensorBuilderSet tensor_builders;
  for (const auto op : _operations)
  {
    const auto tensor_builder = op->lower_info()->backend().stage_gen()->tensor_builder();
    for (const auto &ind : op->getInputs())
    {
      tensor_builder->mark(ind);
      tensor_builders.insert(tensor_builder);
    }
    for (const auto &ind : op->getOutputs())
    {
      tensor_builder->mark(ind);
      tensor_builders.insert(tensor_builder);
    }
  }
  return tensor_builders;
}

} // namespace linear
} // namespace neurun
