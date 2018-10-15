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

#ifndef __NEURUN_TEST_GRAPH_OPERATION_MOCK_NODE_H__
#define __NEURUN_TEST_GRAPH_OPERATION_MOCK_NODE_H__

#include "graph/operation/Node.h"
#include "graph/operand/IndexSet.h"

namespace neurun_test
{
namespace graph
{
namespace operation
{

class SimpleMockNode : public neurun::graph::operation::Node
{
public:
  SimpleMockNode(const neurun::graph::operand::IndexSet &inputs,
                 const neurun::graph::operand::IndexSet &outputs)
  {
    setInputs(inputs);
    setOutputs(outputs);
  }

public:
  virtual void accept(neurun::graph::operation::NodeVisitor &&) const override {}
};

} // namespace operation
} // namespace graph
} // namespace neurun_test

#endif // __NEURUN_TEST_GRAPH_OPERATION_MOCK_NODE_H__
