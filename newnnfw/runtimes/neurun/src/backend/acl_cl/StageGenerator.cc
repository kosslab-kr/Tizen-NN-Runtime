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

#include "backend/acl_cl/StageGenerator.h"

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>
#include <arm_compute/runtime/CL/functions/CLPoolingLayer.h>
#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>
#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>
#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h>
#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>

#include "kernel/acl_cl/ConcatLayer.h"

#include "internal/Padding.h"

#include "graph/operand/Index.h"

#include "logging.h"

#include "NeuralNetworks.h"

#include "support/nnapi/Utils.h"

template <typename T> std::unique_ptr<T> make_layer(void) { return std::unique_ptr<T>{new T}; }

::arm_compute::PadStrideInfo asPadStringInfo(const ::internal::Padding &padding,
                                             const ::internal::Stride &stride)
{
  return ::arm_compute::PadStrideInfo{stride.horizontal,
                                      stride.vertical,
                                      padding.left,
                                      padding.right,
                                      padding.top,
                                      padding.bottom,
                                      ::arm_compute::DimensionRoundingType::FLOOR};
}

namespace neurun
{
namespace backend
{
namespace acl_cl
{

//
// ActivationBuilder
//
class ActivationBuilder
{
public:
  ActivationBuilder(IExecutionBuilder &builder) : _builder(builder)
  {
    // DO NOTHING
  }

private:
  void appendReLU(::arm_compute::ICLTensor *tensor);

public:
  void append(FuseCode code, ::arm_compute::ICLTensor *tensor);

private:
  IExecutionBuilder &_builder;
};

void ActivationBuilder::appendReLU(::arm_compute::ICLTensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

  auto fn = make_layer<::arm_compute::CLActivationLayer>();

  fn->configure(ifm_alloc, nullptr, act_info);

  _builder.append(std::move(fn));
}

void ActivationBuilder::append(FuseCode code, ::arm_compute::ICLTensor *ifm_alloc)
{
  switch (code)
  {
    case ANEURALNETWORKS_FUSED_NONE:
    {
      // DO NOTHING
      break;
    }
    case ANEURALNETWORKS_FUSED_RELU:
    {
      appendReLU(ifm_alloc);
      break;
    }
    default:
    {
      throw std::runtime_error("Not supported, yet");
    }
  }
}

//
// StageGenerator
//
StageGenerator::StageGenerator(const neurun::graph::operand::Set &ctx,
                               const std::shared_ptr<TensorBuilder> &tensor_builder)
    : _ctx(ctx), _tensor_builder(tensor_builder)
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

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asKernel();

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

    ::internal::Padding padding;
    ::internal::Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;
  param.padding =
      (padding_type == ANEURALNETWORKS_PADDING_SAME)
          ? ::internal::same_padding(ifm_shape, ofm_shape, stride, ker_shape.W, ker_shape.H)
          : ::internal::valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto ofm_alloc = tensors->at(::neurun::graph::operand::Index{param.ofm_index}).get();
    auto ifm_alloc = tensors->at(::neurun::graph::operand::Index{param.ifm_index}).get();
    auto ker_alloc = tensors->at(::neurun::graph::operand::Index{param.ker_index}).get();
    auto bias_alloc = tensors->at(::neurun::graph::operand::Index{param.bias_index}).get();

    const auto conv_info = asPadStringInfo(param.padding, param.stride);

    std::unique_ptr<::arm_compute::CLConvolutionLayer> fn{new ::arm_compute::CLConvolutionLayer};

    fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info);

    builder.append(std::move(fn));

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };
}

Stage StageGenerator::generate(const graph::operation::MaxPool2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const ::neurun::graph::operand::Index kh_index{node.param().kh_index};
  const ::neurun::graph::operand::Index kw_index{node.param().kw_index};

  const ::neurun::graph::operand::Index vstride_index{node.param().vstride_index};
  const ::neurun::graph::operand::Index hstride_index{node.param().hstride_index};

  const ::neurun::graph::operand::Index padding_index{node.param().padding_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

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

    ::internal::Padding padding;
    ::internal::Stride stride;

    // TODO Add 'activation' field
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? ::internal::same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : ::internal::valid_padding();

  VERBOSE(MaxPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
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

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStringInfo(param.padding, param.stride)};

    std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

    fn->configure(ifm_alloc, ofm_alloc, info);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::AvgPool2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index ifm_index{node.getInputs().at(0)};

  const ::neurun::graph::operand::Index kh_index{node.param().kh_index};
  const ::neurun::graph::operand::Index kw_index{node.param().kw_index};

  const ::neurun::graph::operand::Index vstride_index{node.param().vstride_index};
  const ::neurun::graph::operand::Index hstride_index{node.param().hstride_index};

  const ::neurun::graph::operand::Index padding_index{node.param().padding_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

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

    ::internal::Padding padding;
    ::internal::Stride stride;

    // TODO Add 'activation' field
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? ::internal::same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : ::internal::valid_padding();

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
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

    ::arm_compute::PoolingLayerInfo info{
        ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{param.kw, param.kh},
        asPadStringInfo(param.padding, param.stride), true /* exclude_padding */};

    std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

    fn->configure(ifm_alloc, ofm_alloc, info);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Concat::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index axis_index{node.param().axis_index};

  struct Param
  {
    int32_t output_index;
    std::vector<int32_t> input_indexes;

    int32_t axis;
  };

  Param param;

  param.output_index = ofm_index.asInt();
  for (const auto &e : node.getInputs())
  {
    param.input_indexes.emplace_back(e.asInt());
  }
  param.axis = _ctx.at(axis_index).asScalar<int32_t>();

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();

    std::vector<::arm_compute::ICLTensor *> input_allocs;
    for (auto ifm_ind : param.input_indexes)
    {
      input_allocs.emplace_back(tensors->at(::neurun::graph::operand::Index{ifm_ind}).get());
    }

    std::unique_ptr<::neurun::kernel::acl_cl::ConcatLayer> fn{
        new ::neurun::kernel::acl_cl::ConcatLayer};

    fn->configure(input_allocs, param.axis, output_alloc);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::FullyConnected::Node &node)
{
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

    FuseCode activation;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.weight_index = weight_index.asInt();
  param.bias_index = bias_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();
    auto weight_alloc = tensors->at(::neurun::graph::operand::Index{param.weight_index}).get();
    auto bias_alloc = tensors->at(::neurun::graph::operand::Index{param.bias_index}).get();

    auto fn = make_layer<::arm_compute::CLFullyConnectedLayer>();

    fn->configure(input_alloc, weight_alloc, bias_alloc, output_alloc);

    builder.append(std::move(fn));

    ActivationBuilder{builder}.append(param.activation, output_alloc);
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
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();

    auto fn = make_layer<::arm_compute::CLReshapeLayer>();

    fn->configure(input_alloc, output_alloc);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Softmax::Node &node)
{
  const ::neurun::graph::operand::Index output_index{node.getOutputs().at(0)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};
  const ::neurun::graph::operand::Index scale_index{node.param().scale_index};

  assert(_ctx.at(scale_index).shape().rank() == 0);

  struct Param
  {
    int output_index;
    int input_index;
    float scale;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.scale = _ctx.at(scale_index).asScalar<float>();

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto output_alloc = tensors->at(::neurun::graph::operand::Index{param.output_index}).get();
    auto input_alloc = tensors->at(::neurun::graph::operand::Index{param.input_index}).get();

    auto fn = make_layer<::arm_compute::CLSoftmaxLayer>();

    fn->configure(input_alloc, output_alloc, param.scale);

    builder.append(std::move(fn));
  };
}

Stage StageGenerator::generate(const graph::operation::Add::Node &node)
{
  const ::neurun::graph::operand::Index ofm_index{node.getOutputs().at(0)}
  const ::neurun::graph::operand::Index lhs_index{node.getInputs().at(0)}
  const ::neurun::graph::operand::Index rhs_index{node.getInputs().at(1)}
  const ::neurun::graph::operand::Index activation_index{node.param().activation_index};

  struct Param
  {
    int ofm_index;

    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.output_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto tensors = _tensor_builder;

  return [tensors, param](IExecutionBuilder &builder) {
    auto ofm_alloc = tensors->at(::neurun::graph::operand::Index{param.ofm_index}).get();
    auto lhs_alloc = tensors->at(::neurun::graph::operand::Index{param.lhs_index}).get();
    auto rhs_alloc = tensors->at(::neurun::graph::operand::Index{param.rhs_index}).get();

    auto fn = make_layer<::arm_compute::CLAddLayer>();

    fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

    builder.append(std::move(fn));

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };
}
}


Stage StageGenerator::generate(const graph::operation::NOP::Node & /* node */)
{
  // DO NOTHING
  return nullptr;
}

} // namespace acl_cl
} // namespace backend
} // namespace neurun
