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

#ifndef __NEURUN_GRAPH_DUMPER_H__
#define __NEURUN_GRAPH_DUMPER_H__

#include "graph/operation/NodeVisitor.h"

namespace neurun
{
namespace graph
{
namespace dumper
{

class Dumper : public graph::operation::NodeVisitor
{
public:
  Dumper() = default;

public:
  void visit(const graph::operation::Conv2D::Implicit::Node &node) override;
  void visit(const graph::operation::MaxPool2D::Implicit::Node &node) override;
  void visit(const graph::operation::AvgPool2D::Implicit::Node &node) override;
  void visit(const graph::operation::Concat::Node &node) override;
  void visit(const graph::operation::FullyConnected::Node &node) override;
  void visit(const graph::operation::Reshape::Node &node) override;
  void visit(const graph::operation::Softmax::Node &node) override;
  void visit(const graph::operation::NOP::Node &node) override;
  void visit(const graph::operation::Permute::Node &node) override;
};

} // namespace dumper
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_DUMPER_H__
