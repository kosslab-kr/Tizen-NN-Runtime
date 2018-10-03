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

#include "IVerifier.h"

#include "graph/Graph.h"

namespace neurun
{
namespace graph
{
namespace verifier
{

bool DAGChecker::verify(const Graph &graph) const
{
  auto &operations = graph.operations();
  bool cyclic = false;
  std::vector<bool> visited(operations.size(), false);
  std::vector<bool> on_stack(operations.size(), false);

  std::function<void(const operation::Index &index, const operation::Node &)> dfs_recursive =
      [&](const operation::Index &index, const operation::Node &node) -> void {
    if (on_stack[index.value()])
      cyclic = true;
    if (visited[index.value()])
      return;
    visited[index.value()] = true;
    on_stack[index.value()] = true;

    auto outputs = node.getOutputs();
    for (auto output : outputs)
    {
      // TODO Fix traversing algorithm
      //      Every time need to search for operations that has `outgoing` as incoming from all
      //      operations but we can hold that info cached
      operations.iterate([&](const operation::Index &cand_index, const operation::Node &cand_node) {
        auto inputs = cand_node.getInputs();
        for (auto input : inputs)
        {
          if (output == input)
          {
            dfs_recursive(cand_index, cand_node);
          }
        }
      });
    }

    on_stack[index.value()] = false;
  };

  operations.iterate(dfs_recursive);

  return !cyclic;
}

} // namespace verifier
} // namespace graph
} // namespace neurun
