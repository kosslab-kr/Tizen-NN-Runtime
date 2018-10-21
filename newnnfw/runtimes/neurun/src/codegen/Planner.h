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

#ifndef __NEURUN_CODEGEN_PLANNER_H__
#define __NEURUN_CODEGEN_PLANNER_H__

#include "graph/operation/NodeVisitor.h"

namespace neurun
{
namespace graph
{
namespace operand
{
class Set;
} // namespace operand
} // namespace graph
} // namespace neurun

namespace neurun
{
namespace codegen
{

class IPlanBuilder;

class Planner : public graph::operation::NodeVisitor
{
public:
  Planner(const neurun::graph::operand::Set &ctx, neurun::codegen::IPlanBuilder &builder)
      : _ctx{ctx}, _builder{builder}
  {
  }

public:
  virtual void visit(const graph::operation::Conv2D::Implicit::Node &) override;
  virtual void visit(const graph::operation::MaxPool2D::Implicit::Node &) override;
  virtual void visit(const graph::operation::AvgPool2D::Implicit::Node &) override;
  virtual void visit(const graph::operation::Concat::Node &) override;
  virtual void visit(const graph::operation::Reshape::Node &) override;
  virtual void visit(const graph::operation::FullyConnected::Node &) override;
  virtual void visit(const graph::operation::Softmax::Node &) override;
  virtual void visit(const graph::operation::NOP::Node &) override;
  virtual void visit(const graph::operation::Permute::Node &) override;
	virtual void visit(const graph::operation::Tanh::Implicit::Node &) ovdrride;

private:
  const neurun::graph::operand::Set &_ctx;
  neurun::codegen::IPlanBuilder &_builder;
};

} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_PLANNER_H__
