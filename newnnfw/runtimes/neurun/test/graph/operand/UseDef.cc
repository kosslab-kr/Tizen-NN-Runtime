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

#include <gtest/gtest.h>

#include "graph/Graph.h"
#include "graph/verifier/IVerifier.h"
#include "nnfw/std/memory.h"
#include "../operation/MockNode.h"

#include <typeindex>

namespace
{

using IndexSet = neurun::graph::operand::IndexSet;
using MockNode = neurun_test::graph::operation::SimpleMockNode;

} // namespace anonymous

TEST(graph_operand_usedef, usedef_test)
{
  neurun::graph::Graph graph;
  neurun::graph::verifier::DAGChecker verifier;

  neurun::graph::operand::Shape shape{1u};
  neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};
  shape.dim(0) = 3;

  // Model Input/Output
  auto input_operand = graph.addOperand(shape, type);
  auto output_operand = graph.addOperand(shape, type);

  graph.addInput(input_operand);
  graph.operands().at(input_operand).setAsModelInput();
  graph.addOutput(output_operand);
  graph.operands().at(output_operand).setAsOperationOutput();

  // MockNode1
  auto operand_index1 = graph.addOperand(shape, type);
  graph.operands().at(operand_index1).setAsOperationOutput();
  auto mocknode_index1 = graph.addOperation(
      nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand_index1}));

  // MockNode2
  auto operand_index2 = graph.addOperand(shape, type);
  graph.operands().at(operand_index2).setAsOperationOutput();
  auto mocknode_index2 = graph.addOperation(
      nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand_index2}));

  // MockNode3(two input)
  auto multiinput_index = graph.addOperation(nnfw::make_unique<MockNode>(
      IndexSet{operand_index1, operand_index2}, IndexSet{output_operand}));

  graph.finishBuilding();

  ASSERT_EQ(verifier.verify(graph), true);

  const auto &operations = graph.operations();
  // Check def
  ASSERT_EQ(graph.operands().at(operand_index1).getDef().contains(mocknode_index1), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getDef().contains(mocknode_index2), true);
  ASSERT_EQ(graph.operands().at(output_operand).getDef().contains(multiinput_index), true);

  ASSERT_EQ(graph.operands().at(operand_index1).getDef().contains(mocknode_index2), false);
  ASSERT_EQ(graph.operands().at(operand_index1).getDef().contains(multiinput_index), false);

  // Check use
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index1), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index2), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(multiinput_index), false);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().contains(multiinput_index), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getUses().contains(multiinput_index), true);

  ASSERT_EQ(graph.operands().at(input_operand).getUses().size(), 2);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(output_operand).getUses().size(), 0);
}

TEST(graph_operand_usedef, usedef_test_insertion)
{
  neurun::graph::Graph graph;
  neurun::graph::verifier::DAGChecker verifier;

  neurun::graph::operand::Shape shape{1u};
  neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};
  shape.dim(0) = 3;

  // Model Input/Output
  auto input_operand = graph.addOperand(shape, type);
  auto output_operand = graph.addOperand(shape, type);

  graph.addInput(input_operand);
  graph.operands().at(input_operand).setAsModelInput();
  graph.addOutput(output_operand);
  graph.operands().at(output_operand).setAsOperationOutput();

  // MockNode1
  auto operand_index1 = graph.addOperand(shape, type);
  graph.operands().at(operand_index1).setAsOperationOutput();
  auto mocknode_index1 = graph.addOperation(
      nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand_index1}));

  // MockNode2
  auto operand_index2 = graph.addOperand(shape, type);
  graph.operands().at(operand_index2).setAsOperationOutput();
  auto mocknode_index2 = graph.addOperation(
      nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand_index2}));

  // MockNode3(two input)
  auto multiinput_index = graph.addOperation(nnfw::make_unique<MockNode>(
      IndexSet{operand_index1, operand_index2}, IndexSet{output_operand}));

  graph.finishBuilding();

  // Insert node1 (between 1 and 2)
  auto inserted_operand1 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand1).setAsOperationOutput();
  auto inserted_index1 =
      graph.insertOperation(input_operand, mocknode_index2,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand1}));

  ASSERT_EQ(inserted_index1.asInt(), 3);

  // Insert node2 (between 2 and 3)
  auto inserted_operand2 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand2).setAsOperationOutput();
  auto inserted_index2 =
      graph.insertOperation(operand_index2, multiinput_index,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand2}));

  ASSERT_EQ(inserted_index2.asInt(), 4);

  ASSERT_EQ(verifier.verify(graph), true);

  // Check def
  ASSERT_EQ(graph.operands().at(input_operand).getDef().size(), 0);
  ASSERT_EQ(graph.operands().at(operand_index1).getDef().contains(mocknode_index1), true);
  ASSERT_EQ(graph.operands().at(inserted_operand1).getDef().contains(inserted_index1), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getDef().contains(mocknode_index2), true);
  ASSERT_EQ(graph.operands().at(inserted_operand2).getDef().contains(inserted_index2), true);
  ASSERT_EQ(graph.operands().at(output_operand).getDef().contains(multiinput_index), true);

  // Check use
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index1), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(inserted_index1), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index2), false);
  ASSERT_EQ(graph.operands().at(inserted_operand1).getUses().contains(mocknode_index2), true);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().contains(multiinput_index), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getUses().contains(inserted_index2), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getUses().contains(multiinput_index), false);
  ASSERT_EQ(graph.operands().at(inserted_operand2).getUses().contains(multiinput_index), true);

  ASSERT_EQ(graph.operands().at(input_operand).getUses().size(), 2);
  ASSERT_EQ(graph.operands().at(inserted_operand1).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(inserted_operand2).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(operand_index2).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(output_operand).getUses().size(), 0);
}
