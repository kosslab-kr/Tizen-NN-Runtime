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

#include "StageGenerator.h"

#include <stdexcept>

#include "internal/Padding.h"
#include "kernel/cpu/OperationUtils.h"
#include "kernel/cpu/ConvolutionLayer.h"
#include "kernel/cpu/AvgPoolLayer.h"
#include "kernel/cpu/MaxPoolLayer.h"
#include "kernel/cpu/ConcatLayer.h"
#include "kernel/cpu/FullyConnectedLayer.h"
#include "kernel/cpu/ReshapeLayer.h"
#include "kernel/cpu/SoftMaxLayer.h"

#include "logging.h"

#include "support/nnapi/Utils.h"

#include "logging.h"

namespace neurun
{
namespace backend
{
namespace cpu
{

StageGenerator::StageGenerator(const neurun::graph::operand::Set &operand_ctx,
                               const std::shared_ptr<TensorBuilder> &tensor_builder)
    : _ctx(operand_ctx), _tensor_builder(tensor_builder)
{
  // DO NOTHING
}

Stage StageGenerator::generate(const graph::operation::Conv2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index ker_index{node.getInputs().at(1)};
  const ::neurun::graph::operand::Index bias_index{node.getInputs().at(2)};

  const ::neurun::graph::operand::Index vstride_index{node.param().vstride_index};
  const ::neurun::graph::operand::Index hstride_index{node.param().hstride_index};

  const ::neurun::graph::operand::Index padding_index{node.param().padding_index};
  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  ::internal::Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;
    ::neurun::kernel::cpu::Shape ker_shape;
    ::neurun::kernel::cpu::Shape bias_shape;

    ::internal::Padding padding;
    ::internal::Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ofm_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ifm_index));
  param.ker_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ker_index));
  param.bias_shape = ::neurun::kernel::cpu::getShape(_ctx.at(bias_index));

  param.stride = stride;
  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? ::internal::same_padding(_ctx.at(ifm_index).shape().asFeature(),
                                                 _ctx.at(ofm_index).shape().asFeature(), stride,
                                                 _ctx.at(ker_index).shape().asKernel().W,
                                                 _ctx.at(ker_index).shape().asKernel().H)
                      : ::internal::valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto ofm_alloc = tensors->at(::neurun::graph::operand::Index{param.ofm_index});
    auto ifm_alloc = tensors->at(::neurun::graph::operand::Index{param.ifm_index});
    auto ker_alloc = tensors->at(::neurun::graph::operand::Index{param.ker_index});
    auto bias_alloc = tensors->at(::neurun::graph::operand::Index{param.bias_index});

    std::unique_ptr<::neurun::kernel::cpu::ConvolutionLayer> fn{
        new ::neurun::kernel::cpu::ConvolutionLayer};

    fn->configure(ifm_alloc->buffer(), param.ifm_shape, ker_alloc->buffer(), param.ker_shape,
                  bias_alloc->buffer(), param.bias_shape, param.padding.left, param.padding.right,
                  param.padding.top, param.padding.bottom, param.stride.horizontal,
                  param.stride.vertical, param.activation, ofm_alloc->buffer(), param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::MaxPool2D::Implicit::Node &node)
{
  VERBOSE(MaxPool2D) << "generate CPU MaxPool2D" << std::endl;

  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const ::neurun::graph::operand::Index kh_index{node.param().kh_index};
  const ::neurun::graph::operand::Index kw_index{node.param().kw_index};

  const ::neurun::graph::operand::Index vstride_index{node.param().vstride_index};
  const ::neurun::graph::operand::Index hstride_index{node.param().hstride_index};

  const ::neurun::graph::operand::Index padding_index{node.param().padding_index};
  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;

    ::internal::Padding padding;
    ::internal::Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ofm_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ifm_index));

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding =
      (padding_type == ANEURALNETWORKS_PADDING_SAME)
          ? ::internal::same_padding(_ctx.at(ifm_index).shape().asFeature(),
                                     _ctx.at(ofm_index).shape().asFeature(), param.stride, kw, kh)
          : ::internal::valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(MaxPool2D) << "IFM_H: " << _ctx.at(ifm_index).shape().asFeature().H << std::endl;
  VERBOSE(MaxPool2D) << "IFM_W: " << _ctx.at(ifm_index).shape().asFeature().W << std::endl;
  VERBOSE(MaxPool2D) << "OFM_H: " << _ctx.at(ofm_index).shape().asFeature().H << std::endl;
  VERBOSE(MaxPool2D) << "OFM_W: " << _ctx.at(ofm_index).shape().asFeature().W << std::endl;
  VERBOSE(MaxPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(MaxPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(MaxPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(MaxPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(MaxPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(MaxPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto ofm_alloc = tensors->at(::neurun::graph::operand::Index{param.ofm_index}).get();
    auto ifm_alloc = tensors->at(::neurun::graph::operand::Index{param.ifm_index}).get();

    std::unique_ptr<::neurun::kernel::cpu::MaxPoolLayer> fn{
        new ::neurun::kernel::cpu::MaxPoolLayer};

    fn->configure(ifm_alloc->buffer(), param.ifm_shape, param.padding.left, param.padding.right,
                  param.padding.top, param.padding.bottom, param.stride.horizontal,
                  param.stride.vertical, param.kw, param.kh, param.activation, ofm_alloc->buffer(),
                  param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::AvgPool2D::Implicit::Node &node)
{
  VERBOSE(AvgPool2D) << "generate CPU AvgPool2D" << std::endl;

  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const ::neurun::graph::operand::Index kh_index{node.param().kh_index};
  const ::neurun::graph::operand::Index kw_index{node.param().kw_index};

  const ::neurun::graph::operand::Index vstride_index{node.param().vstride_index};
  const ::neurun::graph::operand::Index hstride_index{node.param().hstride_index};

  const ::neurun::graph::operand::Index padding_index{node.param().padding_index};
  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;

    ::internal::Padding padding;
    ::internal::Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ofm_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ifm_index));

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding =
      (padding_type == ANEURALNETWORKS_PADDING_SAME)
          ? ::internal::same_padding(_ctx.at(ifm_index).shape().asFeature(),
                                     _ctx.at(ofm_index).shape().asFeature(), param.stride, kw, kh)
          : ::internal::valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(AvgPool2D) << "IFM_H: " << _ctx.at(ifm_index).shape().asFeature().H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << _ctx.at(ifm_index).shape().asFeature().W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << _ctx.at(ofm_index).shape().asFeature().H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << _ctx.at(ofm_index).shape().asFeature().W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(AvgPool2D) << "PAD: " << ::nnfw::support::nnapi::to_string(padding_type) << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto ofm_alloc = tensors->at(::neurun::graph::operand::Index{param.ofm_index}).get();
    auto ifm_alloc = tensors->at(::neurun::graph::operand::Index{param.ifm_index}).get();

    std::unique_ptr<::neurun::kernel::cpu::AvgPoolLayer> fn{
        new ::neurun::kernel::cpu::AvgPoolLayer};

    fn->configure(ifm_alloc->buffer(), param.ifm_shape, param.padding.left, param.padding.right,
                  param.padding.top, param.padding.bottom, param.stride.horizontal,
                  param.stride.vertical, param.kw, param.kh, param.activation, ofm_alloc->buffer(),
                  param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Concat::Node &node)
{
  VERBOSE(Concat) << "generate CPU Concat" << std::endl;

  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index axis_index{node.param().axis_index};

  struct Param
  {
    int32_t output_index;
    std::vector<int32_t> input_indexes;

    int32_t axis;

    ::neurun::kernel::cpu::Shape ofm_shape;
    std::vector<::neurun::kernel::cpu::Shape> ifm_shapes;
  };

  Param param;

  param.output_index = ofm_index.asInt();
  for (const auto &e : node.getInputs())
  {
    param.input_indexes.emplace_back(e.asInt());
  }
  param.axis = _ctx.at(axis_index).asScalar<int32_t>();

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(ofm_index));

  for (auto e : node.getInputs())
  {
    param.ifm_shapes.emplace_back(::neurun::kernel::cpu::getShape(_ctx.at(e)));
  }

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();

    std::vector<const uint8_t *> input_buffers;
    for (auto ifm_ind : param.input_indexes)
    {
      input_buffers.emplace_back(
          tensors->at(::neurun::graph::operand::Index{ifm_ind}).get()->buffer());
    }

    std::unique_ptr<::neurun::kernel::cpu::ConcatLayer> fn{new ::neurun::kernel::cpu::ConcatLayer};

    fn->configure(input_buffers, param.ifm_shapes, param.axis, output_alloc->buffer(),
                  param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::FullyConnected::Node &node)
{
  VERBOSE(FullyConnected) << "generate CPU FullyConnected" << std::endl;

  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index weight_index{node.getInputs().at(1)};
  const ::neurun::graph::operand::Index bias_index{node.getInputs().at(2)};
  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int weight_index;
    int bias_index;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;
    ::neurun::kernel::cpu::Shape weight_shape;
    ::neurun::kernel::cpu::Shape bias_shape;

    FuseCode activation;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.weight_index = weight_index.asInt();
  param.bias_index = bias_index.asInt();

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(output_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(input_index));
  param.weight_shape = ::neurun::kernel::cpu::getShape(_ctx.at(weight_index));
  param.bias_shape = ::neurun::kernel::cpu::getShape(_ctx.at(bias_index));

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();
    auto weight_alloc = tensors->at(::neurun::graph::operand::Index{param.weight_index}).get();
    auto bias_alloc = tensors->at(::neurun::graph::operand::Index{param.bias_index}).get();

    std::unique_ptr<::neurun::kernel::cpu::FullyConnectedLayer> fn{
        new ::neurun::kernel::cpu::FullyConnectedLayer};

    fn->configure(input_alloc->buffer(), param.ifm_shape, weight_alloc->buffer(),
                  param.weight_shape, bias_alloc->buffer(), param.bias_shape, param.activation,
                  output_alloc->buffer(), param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Reshape::Node &node)
{
  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};

  struct Param
  {
    int output_index;
    int input_index;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(output_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(input_index));

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();

    std::unique_ptr<::neurun::kernel::cpu::ReshapeLayer> fn{
        new ::neurun::kernel::cpu::ReshapeLayer};

    fn->configure(input_alloc->buffer(), param.ifm_shape, output_alloc->buffer(), param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Softmax::Node &node)
{
  VERBOSE(Softmax) << "generate CPU Softmax" << std::endl;

  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index scale_index{node.param().scale_index};

  struct Param
  {
    int output_index;
    int input_index;

    ::neurun::kernel::cpu::Shape ofm_shape;
    ::neurun::kernel::cpu::Shape ifm_shape;

    float scale;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  param.ofm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(output_index));
  param.ifm_shape = ::neurun::kernel::cpu::getShape(_ctx.at(input_index));

  param.scale = _ctx.at(scale_index).asScalar<float>();

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();

    std::unique_ptr<::neurun::kernel::cpu::SoftMaxLayer> fn{
        new ::neurun::kernel::cpu::SoftMaxLayer};

    fn->configure(input_alloc->buffer(), param.ifm_shape, param.scale, output_alloc->buffer(),
                  param.ofm_shape);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::NOP::Node & /* node */)
{
  // DO NOTHING
  return nullptr;
}

} // namespace neurun
} // namespace backend
} // namespace cpu
