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

#include "graph/operation/Node.h"
#include "graph/Graph.h"
#include "graph/verifier/IVerifier.h"
#include "nnfw/std/memory.h"
#include "graph/operand/Object.h"

class MockNode : public neurun::graph::operation::Node
{
public:
  MockNode(const neurun::graph::operand::Index &input, const neurun::graph::operand::Index &output)
  {
    setInputs({input});
    setOutputs({output});
  }

public:
  virtual void accept(neurun::graph::operation::NodeVisitor &&) const override {}
};

TEST(Verifier, dag_checker)
{
  neurun::graph::Graph graph;
  neurun::graph::verifier::DAGChecker verifier;

  ::neurun::graph::operand::Shape shape{1u};
  ::neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};
  shape.dim(0) = 3;

  auto operand1 = graph.addOperand(shape, type);
  auto operand2 = graph.addOperand(shape, type);

  graph.addInput(operand1);
  graph.operands().at(operand1).setAsModelInput();
  graph.addOutput(operand2);
  graph.operands().at(operand2).setAsOperationOutput();

  graph.addOperation(nnfw::make_unique<MockNode>(operand1, operand2));

  ASSERT_EQ(verifier.verify(graph), true);

  // Create cycle
  graph.operands().at(operand1).setAsOperationOutput();
  graph.addOperation(nnfw::make_unique<MockNode>(operand2, operand1));

  ASSERT_EQ(verifier.verify(graph), false);
}
