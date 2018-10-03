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

#include "Conv2D.h"

#include <cassert>

#include "NodeVisitor.h"
#include "LowerInfo.h"

namespace neurun
{
namespace graph
{
namespace operation
{
namespace Conv2D
{
namespace Implicit
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

Node::Node(const graph::operation::Node::InitParam &init_param)
{
  assert(init_param.input_count == 7 && init_param.output_count == 1);

  // Each input should be interpreted as follows:
  //
  //
  //  0 -> IFM Tensor Index
  //  1 -> Kernel Tensor Index
  //  2 -> Bias Tensor Index
  //  3 -> Padding Code (ANEURALNETWORKS_PADDING_SAME or ANEURALNETWORKS_PADDING_VALID) Index
  //  4 -> Stride (width) Index
  //  5 -> Stride (height) INdex
  //  6 -> Activation Index

  setInputs({init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]});
  setOutputs({init_param.outputs[0]});

  _param.padding_index = init_param.inputs[3];
  _param.hstride_index = init_param.inputs[4];
  _param.vstride_index = init_param.inputs[5];
  _param.activation_index = init_param.inputs[6];
}

void Node::setInputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 3);

  graph::operation::Node::setInputs(indexes);
}

void Node::setOutputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 1);

  graph::operation::Node::setOutputs(indexes);
}

} // namespace Implicit
} // namespace Conv2D
} // namespace operation
} // namespace graph
} // namespace neurun
