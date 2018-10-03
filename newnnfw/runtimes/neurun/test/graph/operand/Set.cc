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

#include "graph/operand/Set.h"

TEST(graph_operand_Set, set_test)
{
  neurun::graph::operand::Set set;

  ::neurun::graph::operand::Shape shape0{3};
  shape0.dim(0) = 1;
  shape0.dim(1) = 2;
  shape0.dim(2) = 3;

  ::neurun::graph::operand::Shape shape1{4};
  shape1.dim(0) = 10;
  shape1.dim(1) = 20;
  shape1.dim(2) = 30;
  shape1.dim(3) = 40;

  ::neurun::graph::operand::TypeInfo type{ANEURALNETWORKS_TENSOR_INT32, 0, 0};

  set.append(shape0, type);
  set.append(shape1, type);

  ASSERT_EQ(set.exist(neurun::graph::operand::Index{0u}), true);
  ASSERT_EQ(set.exist(neurun::graph::operand::Index{1u}), true);
  ASSERT_EQ(set.exist(neurun::graph::operand::Index{2u}), false);

  ASSERT_EQ(set.at(neurun::graph::operand::Index{0u}).shape().dim(0), 1);
  ASSERT_EQ(set.at(neurun::graph::operand::Index{0u}).shape().dim(1), 2);
  ASSERT_EQ(set.at(neurun::graph::operand::Index{0u}).shape().dim(2), 3);
}
