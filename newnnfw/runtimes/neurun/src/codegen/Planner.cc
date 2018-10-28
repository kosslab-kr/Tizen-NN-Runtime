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

#include "Planner.h"

#include <typeinfo>

#include "internal/Convert.h"
#include "graph/operand/Set.h"
#include "codegen/IPlanBuilder.h"
#include "graph/operation/LowerInfo.h"

#include "logging.h"

namespace neurun
{
namespace codegen
{

void Planner::visit(const graph::operation::Conv2D::Implicit::Node &node)
{
  const auto ofm_index = node.getOutputs().at(0);

  const auto ifm_index = node.getInputs().at(0);
  const auto ker_index = node.getInputs().at(1);
  const auto bias_index = node.getInputs().at(2);

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index, ::internal::asTensorInfo(ofm_shape));
  _builder.addShapeConstr(ifm_index, ::internal::asTensorInfo(ifm_shape));
  _builder.addShapeConstr(ker_index, ::internal::asTensorInfo(ker_shape));
  _builder.addShapeConstr(bias_index, ::internal::asTensorInfo(bias_size));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Initializers
  auto init_gen = backend.initializer_gen();
  _builder.addInitializer(ker_index, init_gen->generateWeight(node));
  _builder.addInitializer(bias_index, init_gen->generateBias(node));

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::MaxPool2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index, ::internal::asTensorInfo(ofm_shape));
  _builder.addShapeConstr(ifm_index, ::internal::asTensorInfo(ifm_shape));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::AvgPool2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index, ::internal::asTensorInfo(ofm_shape));
  _builder.addShapeConstr(ifm_index, ::internal::asTensorInfo(ifm_shape));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::Concat::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};

  // NOTE This implementation assumes that input and output are a feature
  // TODO Remove this assumption
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();

  // NOTE This implementation assumes concat over feature depth
  // TODO Remove this assumption
  assert(_ctx.at(::neurun::graph::operand::Index{node.param().axis_index}).asScalar<int32_t>() ==
         3);

  // Set Shape Constraints (for output)
  _builder.addShapeConstr(ofm_index, ::internal::asTensorInfo(ofm_shape));

  // Set Shape Constraints (for input)
  for (const auto &index : node.getInputs())
  {
    const ::neurun::graph::operand::Index ifm_index{index};
    const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
    _builder.addShapeConstr(ifm_index, ::internal::asTensorInfo(ifm_shape));
  }

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::FullyConnected::Node &node)
{
  VERBOSE(FullyConnected) << "Configure FULLY_CONNECTED operation" << std::endl;

  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};

  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index weight_index{node.getInputs().at(1)};
  const ::neurun::graph::operand::Index bias_index{node.getInputs().at(2)};

  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  assert(_ctx.at(output_index).shape().rank() == 2);
  const auto output_size = _ctx.at(output_index).shape().dim(1);

  // NOTE We assume that input is a feature map
  // TODO Remove this restriction!
  const auto ifm_shape = _ctx.at(input_index).shape().asFeature();

  assert(_ctx.at(weight_index).shape().rank() == 2);
  const auto num_output = _ctx.at(weight_index).shape().dim(0);
  const auto input_size = _ctx.at(weight_index).shape().dim(1);
  assert(ifm_shape.C * ifm_shape.H * ifm_shape.W == input_size);

  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  // Set Shape Constraints
  _builder.addShapeConstr(output_index, ::internal::asTensorInfo(output_size));
  _builder.addShapeConstr(input_index, ::internal::asTensorInfo(ifm_shape));
  _builder.addShapeConstr(weight_index,
                          ::internal::asTensorInfo(num_output /*H*/, input_size /*W*/));
  _builder.addShapeConstr(bias_index, ::internal::asTensorInfo(bias_size));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Initializers
  auto init_gen = backend.initializer_gen();
  _builder.addInitializer(weight_index, init_gen->generateWeight(node));
  _builder.addInitializer(bias_index, init_gen->generateBias(node));

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::Reshape::Node &node)
{
  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};

  // NOTE The content of a tensor specified by shape_index should be aligned with
  //      output tensor shape
  // TODO Check consistency of ouput shape

  // 'Feature Map' to 'Vector' reshape
  assert(_ctx.at(input_index).shape().rank() == 4);
  assert(_ctx.at(output_index).shape().rank() == 2);
  assert(_ctx.at(output_index).shape().dim(0) == 1);

  const auto ifm_shape = _ctx.at(input_index).shape().asFeature();
  const auto out_size = _ctx.at(output_index).shape().dim(1);

  // NOTE Vector element ordering issue arises when H or W is not 1
  assert(ifm_shape.H == 1);
  assert(ifm_shape.W == 1);
  assert((ifm_shape.C * ifm_shape.H * ifm_shape.W) == out_size);

  _builder.addShapeConstr(output_index, ::internal::asTensorInfo(out_size));
  _builder.addShapeConstr(input_index, ::internal::asTensorInfo(ifm_shape));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::Softmax::Node &node)
{
  VERBOSE(Softmax) << "Configure SOFTMAX operation" << std::endl;

  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};

  assert(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());

  // TODO Support 'feature map' input
  assert(_ctx.at(input_index).shape().rank() == 2);
  assert(_ctx.at(input_index).shape().dim(0) == 1);
  assert(_ctx.at(input_index).shape().dim(0) == _ctx.at(output_index).shape().dim(0));
  assert(_ctx.at(input_index).shape().dim(1) == _ctx.at(output_index).shape().dim(1));

  const uint32_t len = _ctx.at(output_index).shape().dim(1);

  _builder.addShapeConstr(output_index, ::internal::asTensorInfo(len));
  _builder.addShapeConstr(input_index, ::internal::asTensorInfo(len));

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::NOP::Node & /* node */)
{
  // DO NOTHING
  // TODO : It's just for graph manipulation test now, it should be added tensor copy stage later.
}

void Planner::visit(const graph::operation::Add::Implicit::Node &)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};

  const ::neurun::graph::operand::Index lhs_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index rhs_index{node.getInputs().at(1)};

  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  assert(_ctx.at(lhs_index).shape().rank() == _ctx.at(rhs_index).shape().rank())
  assert(_ctx.at(lhs_index).shape().rank() == _ctx.at(output_index).shape().rank())

  assert(_ctx.at(lhs_index).shape().dim(0) == _ctx.at(rhs_index).shape().dim(0));
  assert(_ctx.at(lhs_index).shape().dim(1) == _ctx.at(output_index).shape().dim(1));

  const auto ofm_shape = _ctx.at(ofm_index).shape().asTensor();
  const auto lhs_index = _ctx.at(lhs_index).shape().asTensor();
  const auto rhs_index = _ctx.at(rhs_index).shape().asTensor();
  
  // Set Shape Constraints
  _builder.addShapeConstr(lhs_index, ::internal::asTensorInfo(lhs_shape));
  _builder.addShapeConstr(rhs_index, ::internal::asTensorInfo(rhs_shape));
  _builder.addShapeConstr(ofm_index, ::internal::asTensorInfo(ofm_shape));
  

  // backend
  auto backend = node.lower_info()->backend();

  // Generate Stage
  auto stage_gen = backend.stage_gen();
  _builder.addStage(stage_gen->generate(node));
}

void Planner::visit(const graph::operation::Permute::Node & /* node */) { throw "NYI"; }

} // namespace codegen
} // namespace neurun
