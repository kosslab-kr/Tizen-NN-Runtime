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

#include "Graph.h"

#include <algorithm>
#include <bitset>

#include "logging.h"
#include "verifier/IVerifier.h"
#include "nnfw/std/memory.h"
#include "linear/Linear.h"
#include "operation/LowerInfo.h"
#include "operand/LowerInfo.h"
#include "operand/Shape4DConvert.h"
#include "codegen/BackendResolver.h"
#include "backend/IBackendConfig.h"

namespace neurun
{
namespace graph
{

operand::Index Graph::addOperand(const operand::Shape &shape, const operand::TypeInfo &type)
{
  return _operands.append(shape, type);
}

operation::Index Graph::addOperation(std::unique_ptr<operation::Node> &&node)
{
  assert(_phase == Phase::BUILDING);
  return _operations.append(std::move(node));
}

// TODO : If operand's use-def information is introduced,
//        Following API and implements would be refactored.
/**
 * @brief Insert operation into between an operand and next operation.
 *
 * @param prev_operand_index is an previous operand index of insertion.
 * @param next_operation_index is an next operation index of insertion.
 * @param node is an operation::Node to insert.
 *
 * @return operation::Index
 */
operation::Index Graph::insertOperation(const operand::Index &prev_operand_index,
                                        const operation::Index &next_operation_index,
                                        std::unique_ptr<operation::Node> &&node)
{
  assert(_phase != Phase::BUILDING);
  auto &next_operation = _operations.at(next_operation_index);
  auto next_input_indexes = next_operation.getInputs();

  assert(next_input_indexes.contains(prev_operand_index));
  assert(node->getInputs().size() == 0); // node to be inserted must not have any inputs

  node->setInputs({prev_operand_index});

  // For multi input operation (ex. concat)
  operand::IndexSet index_set;
  auto cur_output_indexes = node->getOutputs();
  assert(cur_output_indexes.size() == 1); // Assume output of inserted node size always 1
  auto cur_output_index = cur_output_indexes.at(operand::IO::Index{0});
  // TODO : If the API for setting input one by one is introduced, it would be changed to simple.
  for (auto next_input_index : next_input_indexes)
  {
    if (prev_operand_index == next_input_index)
    {
      index_set.append(cur_output_index);
    }
    else
    {
      index_set.append(next_input_index);
    }
  }

  next_operation.setInputs(index_set);

  operation::Index node_index = _operations.append(std::move(node));

  // Update Use/Def info
  {
    _operands.at(prev_operand_index).removeUse(next_operation_index);
    _operands.at(cur_output_index).appendUse(next_operation_index);

    _operands.at(prev_operand_index).appendUse(node_index);

    auto node_output_indexes = _operations.at(node_index).getOutputs();
    assert(node_output_indexes.size() == 1);
    auto node_output_index = node_output_indexes.at(operand::IO::Index{0});
    _operands.at(node_output_index).appendDef(node_index);
  }

  return node_index;
}

void Graph::setOperandValue(const operand::Index &ind, std::unique_ptr<operand::Data> &&data)
{
  assert(_phase == Phase::BUILDING);
  assert(_operands.exist(ind));
  _operands.at(ind).data(std::move(data));
}

void Graph::addInput(const operand::Index &ind)
{
  assert(_phase == Phase::BUILDING);
  _inputs.append(ind);
}

void Graph::addOutput(const operand::Index &ind)
{
  assert(_phase == Phase::BUILDING);
  _outputs.append(ind);
}

void Graph::finishBuilding(void)
{
  assert(_phase == Phase::BUILDING);
  _phase = Phase::MODEL;

  // Initialize operand use-def
  initializeUseDef();

  // Call graph verifications for the MODEL phase
  {
    verifier::DAGChecker dag_checker;
    dag_checker.verify(*this);
  }
}

void Graph::lower(void)
{
  assert(_phase == Phase::MODEL);

  // Lower
  {
    // operand::LowerInfo holder
    std::unordered_map<operand::Index, std::unique_ptr<operand::LowerInfo>> operands_lower_info;

    _operands.iterate([&](const operand::Index &index, const operand::Object &object) {
      operands_lower_info[index] =
          nnfw::make_unique<operand::LowerInfo>(operand::asShape4D(object.shape()));
    });

    auto _backend_resolver = codegen::BackendResolver(_operands);

    _operations.iterate([&](const operation::Index &, operation::Node &node) {
      auto backend = _backend_resolver.getBackend(typeid(node));

      // Operation LowerInfo
      node.lower_info(nnfw::make_unique<operation::LowerInfo>(backend));

      // LowerInfo for in/output operands
      for (auto operand : node.getInputs())
      {
        auto &&lower_info = operands_lower_info.at(operand);
        lower_info->addUseLayout(backend.config()->getOperandLayout());
      }
      for (auto operand : node.getOutputs())
      {
        auto &&lower_info = operands_lower_info.at(operand);
        lower_info->addDefLayout(backend.config()->getOperandLayout());
      }
    });

    // Set LowerInfo for each operand from the operand::LowerInfo holder
    _operands.iterate([&](const operand::Index &index, operand::Object &object) {
      object.lower_info(std::move(operands_lower_info[index]));

      // Dump operand LowerInfo
      {
        auto layouts_to_string = [](const operand::LayoutSet &layouts) {
          std::string str;
          for (auto layout : layouts)
          {
            const char *name = "";
            if (layout == operand::Layout::NHWC)
              name = "NHWC";
            if (layout == operand::Layout::NCHW)
              name = "NCHW";
            str += name;
            str += " ";
          }
          return "{ " + str + "}";
        };

        const auto &lower_info = object.lower_info();
        const auto &shape = lower_info->shape();
        std::string def_layouts = layouts_to_string(lower_info->def_layouts());
        std::string use_layouts = layouts_to_string(lower_info->use_layouts());
        VERBOSE(Lower) << "* Operand #" << index.value() << " LowerInfo" << std::endl;
        VERBOSE(Lower) << "  - 4D Shape (NHWC) : { " << shape.n() << " " << shape.h() << " "
                       << shape.w() << " " << shape.c() << " "
                       << "}" << std::endl;
        VERBOSE(Lower) << "  - Def Layout      : " << def_layouts << std::endl;
        VERBOSE(Lower) << "  - Use Layout      : " << use_layouts << std::endl;
      }
    });
  }

  // Graph verifications for the LOWERED phase
  {
    verifier::DAGChecker dag_checker;
    dag_checker.verify(*this);
  }

  _phase = Phase::LOWERED;
}

std::unique_ptr<linear::Linear> Graph::linearize(void)
{
  assert(_phase == Phase::LOWERED);

  auto linear = nnfw::make_unique<linear::Linear>(*this);

  // TODO Move the operations and operands to linear object

  _phase = Phase::LINEARIZED;

  return std::move(linear);
}

void Graph::initializeUseDef()
{
  operations().iterate([&](const operation::Index &index, const operation::Node &node) -> void {
    auto outputs = node.getOutputs();
    for (auto output : outputs)
    {
      operands().at(output).appendDef(index);
    }

    auto inputs = node.getInputs();
    for (auto input : inputs)
    {
      operands().at(input).appendUse(index);
    }
  });
}

} // namespace graph
} // namespace neurun

namespace neurun
{
namespace graph
{

// Explicit instantiations to have implementation in the source file.

template class Graph::DefaultIterator<true>;
template class Graph::DefaultIterator<false>;

template class Graph::PostDfsIterator<true>;
template class Graph::PostDfsIterator<false>;

//
// Graph::DefaultIterator
//

template <bool is_const>
void Graph::DefaultIterator<is_const>::iterate(GraphRef graph, const IterFn &fn) const
{
  graph._operations.iterate([&](const operation::Index &, NodeRef node) -> void { fn(node); });
}

//
// Graph::PostDfsIterator
//

template <bool is_const>
void Graph::PostDfsIterator<is_const>::iterate(GraphRef graph, const IterFn &fn) const
{
  assert(!graph.isBuildingPhase()); // Restrict iteration condition

  std::vector<bool> visited(graph._operations.size(), false);

  std::function<void(const operation::Index &, NodeRef)> dfs_recursive =
      [&](const operation::Index &index, NodeRef node) -> void {
    if (visited[index.asInt()])
      return;
    visited[index.asInt()] = true;

    for (auto output : node.getOutputs())
    {
      const auto &operand = graph._operands.at(output);
      for (const auto &use : operand.getUses().list())
      {
        dfs_recursive(use, graph._operations.at(use));
      }
    }

    fn(node);
  };

  graph._operations.iterate(dfs_recursive);

  // All of the operations(nodes) must have been visited.
  assert(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }));
}

} // namespace graph
} // namespace neurun
