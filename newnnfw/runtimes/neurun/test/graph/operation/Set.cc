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

#include "MockNode.h"
#include "graph/operation/Set.h"

using neurun::graph::operation::Set;
using neurun::graph::operation::Node;
using neurun::graph::operation::Index;

TEST(graph_operation_Set, operation_test)
{
  Set set;
  set.append(std::unique_ptr<Node>(
      new neurun_test::graph::operation::SimpleMockNode({1, 2, 3, 4}, {5, 6, 7})));
  Index idx{0u};
  ASSERT_EQ(set.at(idx).getInputs().size(), 4);
  ASSERT_EQ(set.at(idx).getOutputs().size(), 3);
}
