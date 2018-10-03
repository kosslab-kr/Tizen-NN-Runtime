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

TEST(Graph, inputs_and_outputs)
{
  ::neurun::graph::Graph graph;

  ::neurun::graph::operand::Index index0{0u};
  ::neurun::graph::operand::Index index1{1u};

  graph.addInput({index0});
  graph.addInput({index1});

  ::neurun::graph::operand::Index index10{10u};
  ::neurun::graph::operand::Index index11{11u};
  ::neurun::graph::operand::Index index12{12u};

  graph.addOutput({index10});
  graph.addOutput({index11});
  graph.addOutput({index12});

  ASSERT_EQ(graph.getInputs().size(), 2);
  ASSERT_EQ(graph.getOutputs().size(), 3);

  ::neurun::graph::operand::IO::Index io_index0{0};
  ::neurun::graph::operand::IO::Index io_index1{1};
  ::neurun::graph::operand::IO::Index io_index2{2};

  ASSERT_EQ(graph.getInputs().at(io_index0), 0);
  ASSERT_EQ(graph.getInputs().at(io_index1), 1);

  ASSERT_EQ(graph.getOutputs().at(io_index0), 10);
  ASSERT_EQ(graph.getOutputs().at(io_index1), 11);
  ASSERT_EQ(graph.getOutputs().at(io_index2), 12);
}
