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

#include "graph/operand/IndexSet.h"

using neurun::graph::operand::Index;
using neurun::graph::operand::IndexSet;

TEST(graph_operand_IndexSet, index_set_test)
{
  IndexSet iset{0, 2, 4, 8};

  ASSERT_EQ(iset.size(), 4);

  iset.append(Index{10});

  ASSERT_EQ(iset.size(), 5);

  neurun::graph::operand::IO::Index index1{1};
  neurun::graph::operand::IO::Index index2{4};

  ASSERT_EQ(iset.at(index1), 2);
  ASSERT_EQ(iset.at(index2), 10);

  ASSERT_TRUE(iset.contains(Index{2}));
  ASSERT_TRUE(iset.contains(Index{10}));
  ASSERT_FALSE(iset.contains(Index{11}));
}
