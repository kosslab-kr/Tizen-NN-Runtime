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
#include "graph/operand/Index.h"
#include "MockNode.h"

#include <typeindex>

using IOIndex = neurun::graph::operand::IO::Index;
using Index = neurun::graph::operand::Index;
using IndexSet = neurun::graph::operand::IndexSet;
using MockNode = neurun_test::graph::operation::SimpleMockNode;

TEST(graph_operation_manipulation, operation_insertion)
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
  auto operand1 = graph.addOperand(shape, type);
  graph.operands().at(operand1).setAsOperationOutput();
  auto mocknode_index1 =
      graph.addOperation(nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand1}));
  // MockNode2
  auto operand2 = graph.addOperand(shape, type);
  graph.operands().at(operand2).setAsOperationOutput();
  auto mocknode_index2 =
      graph.addOperation(nnfw::make_unique<MockNode>(IndexSet{operand1}, IndexSet{operand2}));
  // MockNode3
  auto mocknode_index3 =
      graph.addOperation(nnfw::make_unique<MockNode>(IndexSet{operand2}, IndexSet{output_operand}));

  graph.finishBuilding();

  ASSERT_EQ(verifier.verify(graph), true);

  // Insert node1 (between 1 and 2)
  auto inserted_operand1 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand1).setAsOperationOutput();
  auto inserted_index1 =
      graph.insertOperation(operand1, mocknode_index2,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand1}));

  ASSERT_EQ(inserted_index1.asInt(), 3);

  // Insert node2 (between 2 and 3)
  auto inserted_operand2 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand2).setAsOperationOutput();
  auto inserted_index2 =
      graph.insertOperation(operand2, mocknode_index3,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand2}));

  ASSERT_EQ(inserted_index2.asInt(), 4);

  // Check tensor indexes
  const auto &operations = graph.operations();
  ASSERT_EQ(operations.at(mocknode_index1).getOutputs().at(Index{0}),
            operations.at(inserted_index1).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(inserted_index1).getOutputs().at(Index{0}),
            operations.at(mocknode_index2).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(mocknode_index2).getOutputs().at(Index{0}),
            operations.at(inserted_index2).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(inserted_index2).getOutputs().at(Index{0}),
            operations.at(mocknode_index3).getInputs().at(Index{0}));

  ASSERT_EQ(verifier.verify(graph), true);
}

TEST(graph_operation_manipulation, operation_insertion_multi_input)
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
  auto operand1 = graph.addOperand(shape, type);
  graph.operands().at(operand1).setAsOperationOutput();
  auto mocknode_index1 =
      graph.addOperation(nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand1}));
  // MockNode2
  auto operand2 = graph.addOperand(shape, type);
  graph.operands().at(operand2).setAsOperationOutput();
  auto mocknode_index2 =
      graph.addOperation(nnfw::make_unique<MockNode>(IndexSet{input_operand}, IndexSet{operand2}));
  // MultiInputMockNode
  auto multiinput_index = graph.addOperation(
      nnfw::make_unique<MockNode>(IndexSet{operand1, operand2}, IndexSet{output_operand}));

  graph.finishBuilding();

  ASSERT_EQ(verifier.verify(graph), true);

  // Insert node1 (between 1 and multi)
  auto inserted_operand1 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand1).setAsOperationOutput();
  auto inserted_index1 =
      graph.insertOperation(operand1, multiinput_index,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand1}));

  ASSERT_EQ(inserted_index1.asInt(), 3);

  // Insert node2 (between 2 and multi)
  auto inserted_operand2 = graph.addOperand(shape, type);
  graph.operands().at(inserted_operand2).setAsOperationOutput();
  auto inserted_index2 =
      graph.insertOperation(operand2, multiinput_index,
                            nnfw::make_unique<MockNode>(IndexSet{}, IndexSet{inserted_operand2}));

  ASSERT_EQ(inserted_index2.asInt(), 4);

  // Check tensor indexes
  const auto &operations = graph.operations();
  ASSERT_EQ(operations.at(mocknode_index1).getOutputs().at(Index{0}),
            operations.at(inserted_index1).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(inserted_index1).getOutputs().at(Index{0}),
            operations.at(multiinput_index).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(mocknode_index2).getOutputs().at(Index{0}),
            operations.at(inserted_index2).getInputs().at(Index{0}));
  ASSERT_EQ(operations.at(inserted_index2).getOutputs().at(Index{0}),
            operations.at(multiinput_index).getInputs().at(Index{1}));

  ASSERT_EQ(verifier.verify(graph), true);
}
