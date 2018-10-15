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
#include "nnfw/std/memory.h"
#include "graph/operation/Conv2D.h"
#include "graph/operation/Concat.h"
#include "graph/operand/Index.h"
#include "graph/operand/IndexSet.h"

#include <stdexcept>

using Index = neurun::graph::operand::IO::Index;
using IndexSet = neurun::graph::operand::IndexSet;
using GraphNodeInitParam = neurun::graph::operation::Node::InitParam;

TEST(graph_operation_setIO, operation_setIO_conv)
{
  neurun::graph::Graph graph;

  neurun::graph::operand::Shape shape{1u};
  neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};
  shape.dim(0) = 3;

  // Add Conv
  std::vector<uint32_t> params;
  for (int i = 0; i < 7; ++i)
  {
    params.emplace_back(graph.addOperand(shape, type).asInt());
  }
  uint32_t outoperand = graph.addOperand(shape, type).asInt();

  using GraphNode = neurun::graph::operation::Conv2D::Implicit::Node;

  auto conv = nnfw::make_unique<GraphNode>(GraphNodeInitParam{7, params.data(), 1, &outoperand});
  ASSERT_EQ(conv->getInputs().at(Index{0}).asInt(), params[0]);
  conv->setInputs({8, 9, 10});
  ASSERT_NE(conv->getInputs().at(Index{0}).asInt(), params[0]);
  ASSERT_EQ(conv->getInputs().at(Index{0}).asInt(), 8);
}

TEST(graph_operation_setIO, operation_setIO_concat)
{
  neurun::graph::Graph graph;

  neurun::graph::operand::Shape shape{1u};
  neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};
  shape.dim(0) = 3;

  // Add Concat
  std::vector<uint32_t> params;
  for (int i = 0; i < 7; ++i)
  {
    params.emplace_back(graph.addOperand(shape, type).asInt());
  }
  uint32_t outoperand = graph.addOperand(shape, type).asInt();

  using GraphNode = neurun::graph::operation::Concat::Node;

  auto concat = nnfw::make_unique<GraphNode>(GraphNodeInitParam{7, params.data(), 1, &outoperand});

  ASSERT_EQ(concat->getInputs().size(), 6);
  ASSERT_EQ(concat->getInputs().at(Index{0}).asInt(), params[0]);

  concat->setInputs({80, 6, 9, 11});
  ASSERT_EQ(concat->getInputs().size(), 4);
  ASSERT_NE(concat->getInputs().at(Index{0}).asInt(), params[0]);
  ASSERT_EQ(concat->getInputs().at(Index{0}).asInt(), 80);
  ASSERT_EQ(concat->getInputs().at(Index{2}).asInt(), 9);
  ASSERT_THROW(concat->getInputs().at(Index{5}), std::out_of_range);
}
