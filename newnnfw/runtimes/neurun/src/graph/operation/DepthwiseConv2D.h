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

#ifndef __NEURUN_GRAPH_OPERATION_DEPTHWISE_CONV2D_H__
#define __NEURUN_GRAPH_OPERATION_DEPTIWISE_CONV2D_H__

#include <memory>

#include "graph/operation/Node.h"

namespace neurun
{
namespace graph
{
namespace operation
{
namespace DepthwiseConv2D
{
namespace Implicit
{

struct Param
{
  int32_t hstride_index;
  int32_t vstride_index;

  int32_t padding_index;
  int32_t multipler_index;
  int32_t activation_index;
};

class Node : public graph::operation::Node
{
public:
  Node(const graph::operation::Node::InitParam &);

public:
  virtual void accept(NodeVisitor &&) const override;

public:
  virtual void setInputs(const operand::IndexSet &indexes) override;
  virtual void setOutputs(const operand::IndexSet &indexes) override;

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace Implicit
} // namespace DepthwiseConv2D
} // namespace coperation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_DEPTHWISE_CONV2D_H__
