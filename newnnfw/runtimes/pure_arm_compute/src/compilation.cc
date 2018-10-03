/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <NeuralNetworks.h>

// For CLKernelLibraryEx initialization
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLSubTensor.h>
#include <arm_compute/runtime/CL/functions/CLArithmeticAddition.h>
#include <arm_compute/runtime/CL/functions/CLArithmeticSubtraction.h>
#include <arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h>
#include <arm_compute/runtime/CL/functions/CLPixelWiseDivision.h>
#include <arm_compute/runtime/CL/functions/CLPoolingLayer.h>
#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>
#include <arm_compute/runtime/CL/functions/CLScale.h>
#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>
#include <arm_compute/runtime/CL/functions/CLStridedSlice.h>
#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>
#include <arm_compute/runtime/CL/functions/CLGather.h>
#include <arm_compute/runtime/CL/functions/CLTopKV2.h>
#include <arm_compute/runtime/CL/functions/CLReduceMax.h>
#include <arm_compute/runtime/CL/functions/CLCast.h>
#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>
#include <arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h>
#include <arm_compute/runtime/CL/functions/CLDequantizationLayer.h>
#include <arm_compute/runtime/CL/functions/CLReductionMean.h>
#include <arm_compute/runtime/CL/functions/CLTranspose.h>
#include <arm_compute/runtime/CL/functions/CLRNNLayer.h>
#include <arm_compute/runtime/CL/functions/CLFloor.h>
#include <arm_compute/runtime/CL/functions/CLCopy.h>
#include <arm_compute/runtime/CL/functions/CLNormalizationLayer.h>

#include <arm_compute/runtime/SubTensor.h>
#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFloor.h>
#include <arm_compute/runtime/NEON/functions/NENormalizationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

#include "internal/arm_compute.h"
#include "internal/arm_compute/Cast.h"
#include "internal/arm_compute/matrix/View.h"
#include "internal/arm_compute/kernel/View.h"
#include "internal/nnapi/matrix/Reader.h"
#include "internal/nnapi/kernel/Reader.h"
#include "internal/nnapi/feature/Reader.h"
#include "internal/nnapi/feature/View.h"
#include "internal/nnapi/tensor/Reader.h"
#include "internal/arm_compute/feature/View.h"
#include "internal/arm_compute/tensor/View.h"
#include "internal/layers/GenericReshapeLayer.h"
#include "internal/layers/SimpleArithmeticAddition.h"
#include "internal/layers/SimpleCastLayer.h"
#include "internal/layers/GenericFullyConnectedLayer.h"
#include "internal/layers/PadLayer.h"
#include "internal/layers/SimpleSpaceToDepth.h"
#include "internal/layers/SimpleEmbeddingLookup.h"
#include "internal/layers/SquaredDifferenceOperation.h"

#include "util/matrix/IndexIterator.h"
#include "util/kernel/IndexIterator.h"
#include "util/feature/IndexIterator.h"
#include "util/tensor/IndexIterator.h"

#include <nnfw/std/memory.h>

#include "compilation.h"
#include "model.h"
#include "logging.h"

template <typename T> T from_env(const char *);

template <> bool from_env(const char *s)
{
  if (s == nullptr)
  {
    return false;
  }

  return std::stoi(s) != 0;
}

const char *to_string(const PaddingCode &code)
{
  assert((ANEURALNETWORKS_PADDING_SAME == code) || (ANEURALNETWORKS_PADDING_VALID == code));

  switch (code)
  {
    case ANEURALNETWORKS_PADDING_SAME:
      return "ANEURALNETWORKS_PADDING_SAME";
    case ANEURALNETWORKS_PADDING_VALID:
      return "ANEURALNETWORKS_PADDING_VALID";
  }

  return nullptr;
}

struct Padding
{
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;
};

struct Stride
{
  uint32_t vertical;
  uint32_t horizontal;
};

Padding valid_padding(void)
{
  //
  // ANEURALNETWORKS_PADDING_VALID
  //
  // VALID padding. No padding.
  //
  // When the input size is not evenly divisible by the filter size,
  // the input at the end that could not fill the whole filter tile
  // will simply be ignored.
  //
  Padding padding;

  padding.top = 0;
  padding.bottom = 0;
  padding.left = 0;
  padding.right = 0;

  return padding;
}

Padding same_padding(const nnfw::util::feature::Shape &ifm_shape,
                     const nnfw::util::feature::Shape &ofm_shape, const Stride &stride, uint32_t kw,
                     uint32_t kh)
{
  Padding padding;

  // ANEURALNETWORKS_PADDING_SAME (from NNAPI spec)
  //
  // SAME padding. Padding on both ends are the "same":
  //
  //	padding_to_beginning = total_padding / 2
  //  padding_to_end = (total_padding + 1)/2.
  //
  const int32_t vertical_needed_input = (ofm_shape.H - 1) * stride.vertical + kh;
  const int32_t vertical_total_padding = std::max(0, vertical_needed_input - ifm_shape.H);

  const int32_t horizontal_needed_input = (ofm_shape.W - 1) * stride.horizontal + kw;
  const int32_t horizontal_total_padding = std::max(0, horizontal_needed_input - ifm_shape.W);

  padding.top = vertical_total_padding / 2;
  padding.bottom = (vertical_total_padding + 1) / 2;
  padding.left = horizontal_total_padding / 2;
  padding.right = (horizontal_total_padding + 1) / 2;

  return padding;
}

::arm_compute::PadStrideInfo asPadStringInfo(const Padding &padding, const Stride &stride)
{
  return ::arm_compute::PadStrideInfo{stride.horizontal,
                                      stride.vertical,
                                      padding.left,
                                      padding.right,
                                      padding.top,
                                      padding.bottom,
                                      ::arm_compute::DimensionRoundingType::FLOOR};
}

struct IAllocationContext
{
  virtual ~IAllocationContext() = default;

  virtual ::arm_compute::ITensor *at(const ::internal::tflite::operand::Index &ind) const = 0;
};

#include "internal/IExecutionBuilder.h"

using Initializer = std::function<void(::arm_compute::ITensor &)>;
using Stage = std::function<void(const IAllocationContext &, IExecutionBuilder &)>;

using namespace std::placeholders;

template <typename T>
static void initFeatureTensor(::arm_compute::ITensor &tensor,
                              const nnfw::util::feature::Shape &feature_shape,
                              const uint8_t *feature_base, const size_t feature_size)
{
  const ::internal::nnapi::feature::Reader<T> from{
      feature_shape, reinterpret_cast<const T *>(feature_base), feature_size};
  ::internal::arm_compute::feature::View<T> into{&tensor};

  ::nnfw::util::feature::iterate(feature_shape)
      << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
           const auto value = from.at(batch, ch, row, col);
           into.at(batch, ch, row, col) = value;
         };
}

template <typename T>
static void initVectorTensor(::arm_compute::ITensor &tensor, const uint8_t *vec_base,
                             const size_t vec_size)
{
  for (uint32_t n = 0; n < vec_size; ++n)
  {
    const ::arm_compute::Coordinates coordinate{n};

    T *into = reinterpret_cast<T *>(tensor.ptr_to_element(coordinate));

    const T *from = reinterpret_cast<const T *>(vec_base) + n;
    const auto value = *from;

    *into = value;
  }
}

template <typename T>
static void initTensor3D(::arm_compute::ITensor &tensor,
                         const nnfw::util::tensor::Shape &tensor_shape, const uint8_t *tensor_base,
                         const size_t tensor_size)
{
  const ::internal::nnapi::tensor::Reader<T> from{
      tensor_shape, reinterpret_cast<const T *>(tensor_base), tensor_size};
  ::internal::arm_compute::tensor::View<T> into{&tensor};

  ::nnfw::util::tensor::iterate(tensor_shape) << [&](const nnfw::util::tensor::Index &index_nnapi) {
    ::nnfw::util::tensor::Index index_ACL = ::nnfw::util::tensor::copy_reverse(index_nnapi);
    into.at(index_ACL) = from.at(index_nnapi);
  };
}

template <typename T>
static void initMatrixTensor(::arm_compute::ITensor &tensor,
                             const nnfw::util::matrix::Shape &matrix_shape,
                             const uint8_t *matrix_base, const size_t matrix_size)
{
  const ::internal::nnapi::matrix::Reader<T> from{
      matrix_shape, reinterpret_cast<const T *>(matrix_base), matrix_size};
  ::internal::arm_compute::matrix::View<T> into{&tensor};

  ::nnfw::util::matrix::iterate(matrix_shape) << [&](uint32_t row, uint32_t col) {
    const auto value = from.at(row, col);
    into.at(row, col) = value;
  };
}

template <typename T>
static void initReorderVectorTensor(::arm_compute::ITensor &tensor, const uint8_t *vec_base,
                                    const size_t vec_size)
{
  for (uint32_t n = 0; n < vec_size; ++n)
  {
    const ::arm_compute::Coordinates coordinate{ToARMComputeAxis(vec_size, n).value()};

    T *into = reinterpret_cast<T *>(tensor.ptr_to_element(coordinate));

    const T *from = reinterpret_cast<const T *>(vec_base) + n;
    const auto value = *from;

    *into = value;
  }
}

template <typename T>
static void initKernelTensor(::arm_compute::ITensor &tensor,
                             const nnfw::util::kernel::Shape &kernel_shape,
                             const uint8_t *kernel_base, const size_t kernel_size)
{
  const ::internal::nnapi::kernel::Reader<T> from{
      kernel_shape, reinterpret_cast<const T *>(kernel_base), kernel_size};
  ::internal::arm_compute::kernel::View<T> into{&tensor};

  ::nnfw::util::kernel::iterate(kernel_shape)
      << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
           const auto value = from.at(nth, ch, row, col);
           into.at(nth, ch, row, col) = value;
         };
}

struct IPlanBuilder
{
  virtual ~IPlanBuilder() = default;

  virtual void addShapeConstr(const ::internal::tflite::operand::Index &ind,
                              const ::arm_compute::TensorInfo &info) = 0;
  virtual void addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                                    const ::internal::tflite::operand::Index &base,
                                    const ::arm_compute::Coordinates &offset,
                                    const ::arm_compute::TensorShape &shape,
                                    bool extend_parent = false) = 0;
  virtual void addInitializer(const ::internal::tflite::operand::Index &ind,
                              const Initializer &initializer) = 0;
  virtual void addStage(const Stage &) = 0;
};

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
  void appendReLU(::arm_compute::ITensor *tensor);
  void appendReLU6(::arm_compute::ITensor *tensor);
  void appendReLU1(::arm_compute::ITensor *tensor);
  void appendTanh(::arm_compute::ITensor *tensor);

public:
  void append(FuseCode code, ::arm_compute::ITensor *tensor);

private:
  IExecutionBuilder &_builder;
};

void ActivationBuilder::appendReLU(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU", std::move(fn));
  }
  else
  {
    auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU", std::move(fn));
  }
}

void ActivationBuilder::appendReLU1(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU1", std::move(fn));
  }
  else
  {
    auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU1", std::move(fn));
  }
}

void ActivationBuilder::appendReLU6(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU6", std::move(fn));
  }
  else
  {
    auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU6", std::move(fn));
  }
}

void ActivationBuilder::appendTanh(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("Tanh", std::move(fn));
  }
  else
    throw std::runtime_error("Not supported, yet");
}

void ActivationBuilder::append(FuseCode code, ::arm_compute::ITensor *ifm_alloc)
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
    case ANEURALNETWORKS_FUSED_RELU1:
    {
      appendReLU1(ifm_alloc);
      break;
    }
    case ANEURALNETWORKS_FUSED_RELU6:
    {
      appendReLU6(ifm_alloc);
      break;
    }
    default:
    {
      throw std::runtime_error("Not supported, yet");
    }
  }
}

class Planner : public ::internal::tflite::op::NodeVisitor
{
public:
  Planner(const ::internal::tflite::operand::Set &ctx, IPlanBuilder &builder)
      : _ctx{ctx}, _builder{builder}
  {
    // DO NOTHING
  }

public:
  void visit(const ::internal::tflite::op::Add::Node &node) override;
  void visit(const ::internal::tflite::op::Sub::Node &node) override;
  void visit(const ::internal::tflite::op::Mul::Node &node) override;
  void visit(const ::internal::tflite::op::Div::Node &node) override;
  void visit(const ::internal::tflite::op::Conv2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::Conv2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::DepthwiseConv2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::DepthwiseConv2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::Dequantize::Node &node) override;
  void visit(const ::internal::tflite::op::MaxPool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::MaxPool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::AvgPool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::AvgPool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::Concat::Node &node) override;
  void visit(const ::internal::tflite::op::FullyConnected::Node &node) override;
  void visit(const ::internal::tflite::op::ResizeBilinear::Node &node) override;
  void visit(const ::internal::tflite::op::Reshape::Node &node) override;
  void visit(const ::internal::tflite::op::Squeeze::Node &node) override;
  void visit(const ::internal::tflite::op::Softmax::Node &node) override;
  void visit(const ::internal::tflite::op::StridedSlice::Node &node) override;
  void visit(const ::internal::tflite::op::ReduceMax::Node &node) override;
  void visit(const ::internal::tflite::op::Cast::Node &node) override;
  void visit(const ::internal::tflite::op::TopKV2::Node &node) override;
  void visit(const ::internal::tflite::op::Gather::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU1::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU6::Node &node) override;
  void visit(const ::internal::tflite::op::Tanh::Node &node) override;
  void visit(const ::internal::tflite::op::Logistic::Node &node) override;
  void visit(const ::internal::tflite::op::Mean::Node &node) override;
  void visit(const ::internal::tflite::op::RNN::Node &node) override;
  void visit(const ::internal::tflite::op::Transpose::Node &node) override;
  void visit(const ::internal::tflite::op::LSTM::Node &node) override;
  void visit(const ::internal::tflite::op::Floor::Node &node) override;
  void visit(const ::internal::tflite::op::Split::Node &node) override;
  void visit(const ::internal::tflite::op::RSQRT::Node &node) override;
  void visit(const ::internal::tflite::op::Pad::Node &node) override;
  void visit(const ::internal::tflite::op::SpaceToDepth::Node &node) override;
  void visit(const ::internal::tflite::op::L2Pool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::L2Pool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::EmbeddingLookup::Node &node) override;
  void visit(const ::internal::tflite::op::HashtableLookup::Node &node) override;
  void visit(const ::internal::tflite::op::L2Normalization::Node &node) override;
  void visit(const ::internal::tflite::op::SquaredDifference::Node &node) override;

private:
  const ::internal::tflite::operand::Set &_ctx;
  IPlanBuilder &_builder;
};

void Planner::visit(const ::internal::tflite::op::Add::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  const auto lhs_shape = _ctx.at(lhs_index).shape();
  const auto rhs_shape = _ctx.at(rhs_index).shape();
  auto stage = [param, lhs_shape, rhs_shape](const IAllocationContext &ctx,
                                             IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    // NOTE SimpleArithmeticAddition is quite slow, but may be useful for debugging
    if (from_env<bool>(std::getenv("USE_SIMPLE_ARITHMETIC_ADDITION")))
    {
      // NOTE SimpleArithmeticAddition does not support broadcasting
      assert(lhs_shape == rhs_shape);

      auto l = nnfw::make_unique<SimpleArithmeticAddition>();

      l->configure(lhs_alloc, rhs_alloc, ofm_alloc);

      fn = std::move(l);
    }
    else
    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = nnfw::make_unique<::arm_compute::CLArithmeticAddition>();

        // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
        l->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc),
                     ::arm_compute::ConvertPolicy::SATURATE);

        fn = std::move(l);
      }
      else // NEON
      {
        auto l = nnfw::make_unique<::arm_compute::NEArithmeticAddition>();

        // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
        l->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE);

        fn = std::move(l);
      }
    }

    builder.append("Add", std::move(fn));

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Sub::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLArithmeticSubtraction>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc),
                    ::arm_compute::ConvertPolicy::SATURATE);

      builder.append("Sub", std::move(fn));
    }
    else // NEON
    {
      auto fn = nnfw::make_unique<::arm_compute::NEArithmeticSubtraction>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE);

      builder.append("Sub", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

// TODO: test with scalar*scalar, tensor bigger than 3D (e.g., 4D)
void Planner::visit(const ::internal::tflite::op::Mul::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  if (_ctx.at(ofm_index).scale() > 0)
  {
    assert(_ctx.at(ofm_index).scale() > _ctx.at(lhs_index).scale() * _ctx.at(rhs_index).scale());
  }
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {

    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_input_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_input_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLPixelWiseMultiplication>();

      fn->configure(CAST_CL(lhs_input_alloc), CAST_CL(rhs_input_alloc), CAST_CL(output_alloc),
                    1.0, // scale
                    arm_compute::ConvertPolicy::SATURATE,
                    arm_compute::RoundingPolicy::TO_NEAREST_EVEN);

      builder.append("Mul", std::move(fn));
    }
    else // NEON
    {
      auto fn = nnfw::make_unique<::arm_compute::NEPixelWiseMultiplication>();

      fn->configure(CAST_NE(lhs_input_alloc), CAST_NE(rhs_input_alloc), CAST_NE(output_alloc),
                    1.0, // scale
                    arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_ZERO);

      builder.append("Mul", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, output_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Div::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }

  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLPixelWiseDivision>();

      // TODO Decide scale, overflow_policy, and rounding_policy.
      //      Currently, the default values are used.
      fn->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc));

      builder.append("Div", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Conv2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Set initializer for kernel
  {
    auto ker_base = _ctx.at(ker_index).data().base();
    auto ker_size = _ctx.at(ker_index).data().size();
    auto ker_type = _ctx.at(ker_index).type();

    switch (ker_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initKernelTensor<float>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      {
        auto initializer = std::bind(initKernelTensor<uint8_t>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Set initializer for bias
  {
    auto bias_base = _ctx.at(bias_index).data().base();
    auto bias_type = _ctx.at(bias_index).type();

    switch (bias_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initVectorTensor<float>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_INT32:
      {
        auto initializer = std::bind(initVectorTensor<int32_t>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;
  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, stride, ker_shape.W, ker_shape.H)
                      : valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStringInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLConvolutionLayer> fn{new ::arm_compute::CLConvolutionLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info);

      builder.append("Conv2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEConvolutionLayer> fn{new ::arm_compute::NEConvolutionLayer};

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info);

      builder.append("Conv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Conv2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Set initializer for kernel
  // Workaround for https://github.sec.samsung.net/STAR/nnfw/issues/2319
  if (_ctx.at(ker_index).hasData())
  {
    const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
    auto ker_base = _ctx.at(ker_index).data().base();
    auto ker_size = _ctx.at(ker_index).data().size();
    auto ker_type = _ctx.at(ker_index).type();

    switch (ker_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initKernelTensor<float>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      {
        auto initializer = std::bind(initKernelTensor<uint8_t>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Set initializer for bias
  // See above comment.
  if (_ctx.at(bias_index).hasData())
  {
    const auto bias_size = _ctx.at(bias_index).shape().asVector();
    auto bias_base = _ctx.at(bias_index).data().base();
    auto bias_type = _ctx.at(bias_index).type();

    switch (bias_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initVectorTensor<float>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_INT32:
      {
        auto initializer = std::bind(initVectorTensor<int32_t>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStringInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLConvolutionLayer> fn{new ::arm_compute::CLConvolutionLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info);

      builder.append("Conv2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEConvolutionLayer> fn{new ::arm_compute::NEConvolutionLayer};

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info);

      builder.append("Conv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::DepthwiseConv2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index multipler_index{node.param().multipler_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  auto multiplier = _ctx.at(multipler_index).asScalar<int>();

  assert(ker_shape.C == bias_size);
  assert(ker_shape.C == ifm_shape.C * multiplier);

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  // NOTE DepthwiseConv2D kernel is of shape [1, KER_W, KER_H, IFM_C * MULTIPLIER]
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    int multipler;
    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;
  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, stride, ker_shape.W, ker_shape.H)
                      : valid_padding();

  param.multipler = multiplier;
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(DepthwiseConv2D) << "OFM_C: " << ofm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_W: " << ofm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "IFM_C: " << ifm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_W: " << ifm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "KER_C: " << ker_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_H: " << ker_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_W: " << ker_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "STRIDE_H: " << param.stride.vertical << std::endl;
  VERBOSE(DepthwiseConv2D) << "STRIDE_W: " << param.stride.horizontal << std::endl;

  VERBOSE(DepthwiseConv2D) << "ACTIVATION: " << param.activation << std::endl;

  VERBOSE(DepthwiseConv2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStringInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLDepthwiseConvolutionLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEDepthwiseConvolutionLayer>();

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::DepthwiseConv2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index multipler_index{node.param().multipler_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  auto multiplier = _ctx.at(multipler_index).asScalar<int>();

  assert(ker_shape.C == bias_size);
  assert(ker_shape.C == ifm_shape.C * multiplier);

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  // NOTE DepthwiseConv2D kernel is of shape [1, KER_W, KER_H, IFM_C * MULTIPLIER]
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    int multipler;
    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.multipler = multiplier;
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(DepthwiseConv2D) << "OFM_C: " << ofm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_W: " << ofm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "IFM_C: " << ifm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_W: " << ifm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "KER_C: " << ker_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_H: " << ker_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_W: " << ker_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "STRIDE_H: " << param.stride.vertical << std::endl;
  VERBOSE(DepthwiseConv2D) << "STRIDE_W: " << param.stride.horizontal << std::endl;

  VERBOSE(DepthwiseConv2D) << "ACTIVATION: " << param.activation << std::endl;

  VERBOSE(DepthwiseConv2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStringInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLDepthwiseConvolutionLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEDepthwiseConvolutionLayer>();

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Dequantize::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  assert(_ctx.at(input_index).shape().rank() >= 0 && _ctx.at(input_index).shape().rank() <= 4);
  assert(_ctx.at(input_index).shape() == _ctx.at(output_index).shape());
  assert(_ctx.at(input_index).type() == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
  assert(_ctx.at(output_index).type() == ANEURALNETWORKS_TENSOR_FLOAT32);

  // Set Shape Constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    if (from_env<bool>(std::getenv("USE_SIMPLE_CAST")))
    {
      // Use the CPU version of CAST operation
      auto l = nnfw::make_unique<SimpleCastLayer>();

      l->configure(input_alloc, output_alloc);
      fn = std::move(l);
    }
    else // Use the OpenCL version of CAST operation
    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = nnfw::make_unique<::arm_compute::CLCast>();

        l->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));
        fn = std::move(l);
      }
      else
        throw std::runtime_error("Not supported, yet");
    }

    builder.append("Dequantize", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::MaxPool2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

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

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

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

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStringInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("MaxPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("MaxPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::MaxPool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // TODO 4D tensor (dim(0) !=1 )
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

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

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStringInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("MaxPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("MaxPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::AvgPool2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

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

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(AvgPool2D) << "PAD: " << to_string(padding_type) << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{
        ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{param.kw, param.kh},
        asPadStringInfo(param.padding, param.stride), true /* exclude_padding */};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("AvgPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("AvgPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::AvgPool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // TODO 4D tensor (dim(0) != 1)
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{
        ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{param.kw, param.kh},
        asPadStringInfo(param.padding, param.stride), true /* exclude_padding */};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("AvgPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("AvgPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Concat::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  // NOTE This implementation assumes that inputs and output are a feature
  const auto ofm_shape = _ctx.at(ofm_index).shape();
  uint32_t input_rank = ofm_shape.rank();
  int32_t axis = _ctx.at(axis_index).asScalar<int32_t>();

  // Handle negative axis
  if (axis < 0)
  {
    axis += input_rank;
  }

  // Set Shape Constraints and TensorInfo (for output)
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));

  // Set Shape Constraints and TensorInfo (for input)
  const uint32_t coord_index = ToARMComputeAxis(input_rank, axis).value();
  uint32_t depth = 0;

  ::arm_compute::Coordinates coordinates;
  coordinates.set_num_dimensions(input_rank);

  for (const auto &index : node.param().ifm_indexes)
  {
    const ::internal::tflite::operand::Index ifm_index{index};
    const auto ifm_shape = _ctx.at(ifm_index).shape();

    coordinates[coord_index] = depth;

    _builder.addSubsumptionConstr(ifm_index, ofm_index, coordinates,
                                  asTensorShape(_ctx.at(ifm_index).shape()), true);

    depth += ifm_shape.dim(axis);
  }

  // NOTE Concat has no actual operation!
  // However, dummy stage is added because profiler assumes every operation make a stage.
  auto stage = [](const IAllocationContext &ctx, IExecutionBuilder &builder) {};
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::FullyConnected::Node &node)
{
  VERBOSE(FullyConnected) << "Configure FULLY_CONNECTED operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};

  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index weight_index{node.param().weight_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  assert(_ctx.at(input_index).shape().rank() >= 2);
  assert(_ctx.at(output_index).shape().rank() == 2);
  assert(_ctx.at(weight_index).shape().rank() == 2);
  assert(_ctx.at(bias_index).shape().rank() == 1);

  const auto input_rank = _ctx.at(input_index).shape().rank();
  // TODO Currently we are not handling where the case is that the input's rank is 3.
  // The handling should be added in the future.
  assert(input_rank != 3);

  const auto output_size = _ctx.at(output_index).shape().dim(1);
  assert(_ctx.at(bias_index).shape().dim(0) == output_size);
  assert(_ctx.at(weight_index).shape().dim(0) == output_size);
  const auto batch_size = _ctx.at(output_index).shape().dim(0);
  const auto input_size = _ctx.at(weight_index).shape().dim(1);

  // Check for reshaping input's shape into rank-2
  bool needs_reshape = false;
  internal::tflite::operand::Shape reshape(2);
  if (input_rank == 4)
  {
    nnfw::util::feature::Shape ifm_shape_feature = _ctx.at(input_index).shape().asFeature();
    auto feature_size =
        ifm_shape_feature.N * ifm_shape_feature.C * ifm_shape_feature.H * ifm_shape_feature.W;
    assert(feature_size == batch_size * input_size);

    _builder.addShapeConstr(input_index,
                            asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                         _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                         _ctx.at(input_index).zeroPoint()));

    // for reshaping
    needs_reshape = true;
    reshape.dim(0) = batch_size; /* H */
    reshape.dim(1) = input_size; /* W */
  }
  else if (input_rank == 2)
  {
    auto ifm_shape = _ctx.at(input_index).shape();
    nnfw::util::matrix::Shape ifm_shape_matrix = ifm_shape.asMatrix();
    assert(ifm_shape.dim(0) == batch_size);
    assert(ifm_shape.dim(1) == input_size);

    _builder.addShapeConstr(input_index,
                            asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                         _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                         _ctx.at(input_index).zeroPoint()));
  }

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(weight_index,
                          asTensorInfo(asTensorShape(_ctx.at(weight_index).shape()),
                                       _ctx.at(weight_index).type(), _ctx.at(weight_index).scale(),
                                       _ctx.at(weight_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

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

  auto stage = [param, needs_reshape, reshape](const IAllocationContext &ctx,
                                               IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto weight_alloc = ctx.at(::internal::tflite::operand::Index{param.weight_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    auto fn = nnfw::make_unique<GenericFullyConnectedLayer>();

    fn->configure(input_alloc, weight_alloc, bias_alloc, output_alloc, needs_reshape,
                  asTensorShape(reshape));

    builder.append("FullyConnected", std::move(fn));

    ActivationBuilder{builder}.append(param.activation, output_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ResizeBilinear::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index height_index{node.param().height_index};
  const ::internal::tflite::operand::Index width_index{node.param().width_index};

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  struct Param
  {
    int ofm_index;
    int ifm_index;

    int new_height;
    int new_width;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.new_height = _ctx.at(height_index).asScalar<int32_t>();
  param.new_width = _ctx.at(width_index).asScalar<int32_t>();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLScale>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc),
                    ::arm_compute::InterpolationPolicy::BILINEAR,
                    ::arm_compute::BorderMode::REPLICATE, ::arm_compute::PixelValue(0.f),
                    ::arm_compute::SamplingPolicy::TOP_LEFT);

      builder.append("ResizeBilinear", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Reshape::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // NOTE The content of a tensor specified by shape_index should be aligned with
  //      output tensor shape
  // TODO Check consistency of ouput shape

  // TODO Re-enable this assert
  // assert((ifm_shape.C * ifm_shape.H * ifm_shape.W) == out_size);

  // TODO Should move to the place where the operand is handled, if it is possible.
  _builder.addShapeConstr(output_index, asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                                     _ctx.at(output_index).type()));
  _builder.addShapeConstr(input_index, asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                                    _ctx.at(input_index).type()));

  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      // GenericReshape first apply NCHW->NHWC permutation, and apply reshape
      auto fn = nnfw::make_unique<GenericReshapeLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("Reshape", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<GenericReshapeLayer>();

      fn->configure(input_alloc, output_alloc);

      builder.append("Reshape", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Squeeze::Node &node)
{
  // node.param().dims_index_optional is ignored since output tensor already has squeezed shape
  // by freezer and toco
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Currently, 3D-input with dims is tested. Note that param(). dims_index_optional is optional.
  // two generated test passed:
  //   - 3D input : squeeze_float_1
  //   - 2D input : squeeze_3D_float_1
  //   - 4D input fails (squeeze.mod.py) -> we need general tensor support

  // TODO Support generic tensor shape

  // Set Shape Constraints
  _builder.addShapeConstr(output_index, asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                                     _ctx.at(output_index).type()));
  _builder.addShapeConstr(input_index, asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                                    _ctx.at(input_index).type()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLReshapeLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("Squeeze", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEReshapeLayer>();

      fn->configure(input_alloc, output_alloc);

      builder.append("Squeeze", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Softmax::Node &node)
{
  VERBOSE(Softmax) << "Configure SOFTMAX operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index scale_index{node.param().scale_index};

  assert(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
  assert(_ctx.at(scale_index).shape().rank() == 0);

  // TODO Should move to the place where the operand is handled, if it is possible.
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

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

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLSoftmaxLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), param.scale);

      builder.append("Softmax", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NESoftmaxLayer>();

      fn->configure(input_alloc, output_alloc, param.scale);

      builder.append("Softmax", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::StridedSlice::Node &node)
{
  VERBOSE(StridedSlice) << "Configure STRIDED_SLICE operation" << std::endl;

  const ::internal::tflite::operand::Index outputData_index{node.param().outputData_index};

  const ::internal::tflite::operand::Index inputData_index{node.param().inputData_index};
  const ::internal::tflite::operand::Index startData_index{node.param().startData_index};
  const ::internal::tflite::operand::Index endData_index{node.param().endData_index};
  const ::internal::tflite::operand::Index stridesData_index{node.param().stridesData_index};
  const ::internal::tflite::operand::Index beginMask_index{node.param().beginMask_index};
  const ::internal::tflite::operand::Index endMask_index{node.param().endMask_index};
  const ::internal::tflite::operand::Index shrinkAxisMask_index{node.param().shrinkAxisMask_index};

  // Set Shape Constraints
  _builder.addShapeConstr(outputData_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputData_index).shape()),
                                       _ctx.at(outputData_index).type(),
                                       _ctx.at(outputData_index).scale(),
                                       _ctx.at(outputData_index).zeroPoint()));
  _builder.addShapeConstr(
      inputData_index,
      asTensorInfo(asTensorShape(_ctx.at(inputData_index).shape()), _ctx.at(inputData_index).type(),
                   _ctx.at(inputData_index).scale(), _ctx.at(inputData_index).zeroPoint()));

  assert(_ctx.at(startData_index).shape().rank() == 1);
  assert(_ctx.at(endData_index).shape().rank() == 1);
  assert(_ctx.at(stridesData_index).shape().rank() == 1);
  _builder.addShapeConstr(startData_index,
                          asTensorInfo(asTensorShape(_ctx.at(startData_index).shape()),
                                       _ctx.at(startData_index).type()));
  _builder.addShapeConstr(endData_index, asTensorInfo(asTensorShape(_ctx.at(endData_index).shape()),
                                                      _ctx.at(endData_index).type()));
  _builder.addShapeConstr(stridesData_index,
                          asTensorInfo(asTensorShape(_ctx.at(endData_index).shape()),
                                       _ctx.at(stridesData_index).type()));

  // Set initializers for indices data such as order of inputData
  {
    auto startData_base = _ctx.at(startData_index).data().base();
    auto endData_base = _ctx.at(endData_index).data().base();
    auto stridesData_base = _ctx.at(stridesData_index).data().base();
    const auto startData_size = _ctx.at(startData_index).shape().asVector();
    const auto endData_size = _ctx.at(endData_index).shape().asVector();
    const auto stridesData_size = _ctx.at(stridesData_index).shape().asVector();

    assert(_ctx.at(startData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    auto startData_initializer =
        std::bind(initReorderVectorTensor<int32_t>, _1, startData_base, startData_size);
    _builder.addInitializer(startData_index, startData_initializer);

    assert(_ctx.at(endData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    auto endData_initializer =
        std::bind(initReorderVectorTensor<int32_t>, _1, endData_base, endData_size);
    _builder.addInitializer(endData_index, endData_initializer);

    assert(_ctx.at(stridesData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    auto stridesData_initializer =
        std::bind(initReorderVectorTensor<int32_t>, _1, stridesData_base, stridesData_size);
    _builder.addInitializer(stridesData_index, stridesData_initializer);
  }

  struct Param
  {
    int32_t outputData_index;
    int32_t inputData_index;

    int32_t startData_index;
    int32_t endData_index;
    int32_t stridesData_index;

    int32_t beginMask;
    int32_t endMask;
    int32_t shrinkAxisMask;
  };

  Param param;
  param.outputData_index = outputData_index.asInt();
  param.inputData_index = inputData_index.asInt();

  param.startData_index = startData_index.asInt();
  param.endData_index = endData_index.asInt();
  param.stridesData_index = stridesData_index.asInt();

  // Set mask bits such as order of inputData
  const auto inputData_rank = _ctx.at(inputData_index).shape().rank();
  param.beginMask = _ctx.at(beginMask_index).asReorderBits<int32_t>(inputData_rank);
  param.endMask = _ctx.at(endMask_index).asReorderBits<int32_t>(inputData_rank);
  param.shrinkAxisMask = _ctx.at(shrinkAxisMask_index).asReorderBits<int32_t>(inputData_rank);

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto outputData_alloc = ctx.at(::internal::tflite::operand::Index{param.outputData_index});
    auto inputData_alloc = ctx.at(::internal::tflite::operand::Index{param.inputData_index});

    auto startData_alloc = ctx.at(::internal::tflite::operand::Index{param.startData_index});
    auto endData_alloc = ctx.at(::internal::tflite::operand::Index{param.endData_index});
    auto stridesData_alloc = ctx.at(::internal::tflite::operand::Index{param.stridesData_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLStridedSlice>();

      fn->configure(CAST_CL(inputData_alloc), CAST_CL(outputData_alloc), CAST_CL(startData_alloc),
                    CAST_CL(endData_alloc), CAST_CL(stridesData_alloc), param.beginMask,
                    param.endMask, param.shrinkAxisMask);

      builder.append("StridedSlice", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReduceMax::Node &node)
{
  VERBOSE(ReduceMax) << "Configure REDUCEMAX operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  // Handle special case only:
  //   Input: Matrix (rank 2)
  //   Output: Vector (rank 1)
  //   Axis: one element (scalar or rank 1 with 1 element), constant
  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();
  auto axis_shape = _ctx.at(axis_index).shape();
  assert(ofm_shape.rank() == 1);
  assert(ifm_shape.rank() == 2);
  assert(_ctx.at(axis_index).hasData());
  assert(axis_shape.rank() == 0 || ((axis_shape.rank() == 1) && (axis_shape.dim(0) == 1)));

  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Note: Assume only one element in axis. It is checked by assertion above
  // TODO: handle general case
  // Axis is integer value (generally, int32)
  int32_t axis_value = _ctx.at(axis_index).asScalar<int32_t>();
  assert(axis_value == 1);

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    int32_t axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = axis_value;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLReduceMax>();

      fn->configure(CAST_CL(ifm_alloc), param.axis, CAST_CL(ofm_alloc));

      builder.append("ReduceMax", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Cast::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());

  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int input_index;
    int output_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    if (from_env<bool>(std::getenv("USE_SIMPLE_CAST")))
    {
      // Use the CPU version of CAST operation
      auto l = nnfw::make_unique<SimpleCastLayer>();

      l->configure(input_alloc, output_alloc);
      fn = std::move(l);
    }
    else // Use the OpenCL version of CAST operation
    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = nnfw::make_unique<::arm_compute::CLCast>();

        l->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));
        fn = std::move(l);
      }
      else
        throw std::runtime_error("Not supported, yet");
    }

    builder.append("Cast", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::TopKV2::Node &node)
{
  const ::internal::tflite::operand::Index outputValues_index{node.param().outputValues_index};
  const ::internal::tflite::operand::Index outputIndices_index{node.param().outputIndices_index};

  const ::internal::tflite::operand::Index inputData_index{node.param().inputData_index};
  const ::internal::tflite::operand::Index k_index{node.param().k_index};

  // Currently, we only support the vector input.
  assert(_ctx.at(inputData_index).shape().rank() == 1 ||
         _ctx.at(inputData_index).shape().rank() == 2);

  const int32_t k = _ctx.at(k_index).asScalar<int32_t>();

  // Set shape constraints
  _builder.addShapeConstr(outputValues_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputValues_index).shape()),
                                       _ctx.at(outputValues_index).type()));
  _builder.addShapeConstr(outputIndices_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputIndices_index).shape()),
                                       _ctx.at(outputIndices_index).type()));
  _builder.addShapeConstr(inputData_index,
                          asTensorInfo(asTensorShape(_ctx.at(inputData_index).shape()),
                                       _ctx.at(inputData_index).type()));

  // Construct operation parameters
  struct Param
  {
    int32_t outputValues_index;
    int32_t outputIndices_index;

    int32_t inputData_index;
    int32_t k;
  };

  Param param;

  param.outputValues_index = outputValues_index.asInt();
  param.outputIndices_index = outputIndices_index.asInt();
  param.inputData_index = inputData_index.asInt();
  param.k = k;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto values_alloc = ctx.at(::internal::tflite::operand::Index{param.outputValues_index});
    auto indices_alloc = ctx.at(::internal::tflite::operand::Index{param.outputIndices_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.inputData_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLTopKV2>();

      fn->configure(CAST_CL(input_alloc), param.k, CAST_CL(values_alloc), CAST_CL(indices_alloc));

      builder.append("TopKV2", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Gather::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};

  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  // Currently, 1D-input and 2D-input are supported.
  assert(_ctx.at(lhs_index).shape().rank() == 1 || _ctx.at(lhs_index).shape().rank() == 2);
  assert(_ctx.at(rhs_index).shape().rank() == 1);

  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()),
                                                  _ctx.at(lhs_index).type()));
  _builder.addShapeConstr(rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()),
                                                  _ctx.at(rhs_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    int axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.axis = static_cast<int>(_ctx.at(axis_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::IFunction> fn;

      auto l = nnfw::make_unique<::arm_compute::CLGather>();
      l->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc));
      fn = std::move(l);
      builder.append("Gather", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU::Node &node)
{
  VERBOSE(ReLU) << "Configure ReLU operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU1::Node &node)
{
  VERBOSE(ReLU1) << "Configure ReLU1 operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU1", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU1", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU6::Node &node)
{
  VERBOSE(ReLU6) << "Configure ReLU6 operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU6", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU6", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Tanh::Node &node)
{
  VERBOSE(Tanh) << "Configure Tanh operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("Tanh", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("Tanh", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Logistic::Node &node)
{
  VERBOSE(Logistic) << "Configure Logistic operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("Logistic", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

// Reduce Mean
void Planner::visit(const ::internal::tflite::op::Mean::Node &node)
{
  VERBOSE(Mean) << "Configure Mean operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};
  const ::internal::tflite::operand::Index keep_dims_index{node.param().keep_dims_index};
  const int keep_dims = _ctx.at(keep_dims_index).asScalar<int>();

  // Set shape constraints
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));
  _builder.addShapeConstr(axis_index, asTensorInfo(asTensorShape(_ctx.at(axis_index).shape()),
                                                   _ctx.at(axis_index).type()));

  // TODO keep_dims==0
  assert(keep_dims != 0);

  // Set axis
  // TODO Other axis (Axis for width and height are currently supported.)
  // TODO Other ranks (Rank 4 is currently supported.)
  assert(_ctx.at(ifm_index).shape().rank() == 4);

  std::vector<uint32_t> axis;
  {
    const auto axis_base = _ctx.at(axis_index).data().base();
    const auto axis_type = _ctx.at(axis_index).type();
    const auto axis_size = _ctx.at(axis_index).shape().asVector();

    // NHWC type -> WHCN type
    if (_ctx.at(ofm_index).shape().rank() == 4)
    {
      for (uint32_t n = 0; n < axis_size; ++n)
      {
        const ::arm_compute::Coordinates coordinate{n};
        const int32_t *from = reinterpret_cast<const int32_t *>(axis_base) + n;
        if (*from == 1)
        {
          axis.push_back(1); // h
        }
        else if (*from == 2)
        {
          axis.push_back(0); // w
        }
        else if (*from < 0)
        {
          // Nothing to do
        }
        else
        {
          throw std::runtime_error{"Not supported axis"};
        }
      }
    }
  }

  struct Param
  {
    int ofm_index;
    int ifm_index;
    std::vector<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLReductionMean>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.axis);

      builder.append("Mean", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::RNN::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index hidden_state_out_index{
      node.param().hidden_state_out_index};

  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index weights_index{node.param().weights_index};
  const ::internal::tflite::operand::Index recurrent_weights_index{
      node.param().recurrent_weights_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};
  const ::internal::tflite::operand::Index hidden_state_in_index{
      node.param().hidden_state_in_index};
  const ::internal::tflite::operand::Index fused_activation_index{
      node.param().fused_activation_index};

  assert(_ctx.at(output_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_out_index).shape().rank() == 2 &&
         _ctx.at(input_index).shape().rank() == 2 && _ctx.at(weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_weights_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_in_index).shape().rank() == 2);
  assert(_ctx.at(bias_index).shape().rank() == 1);

  const auto batch_size = _ctx.at(output_index).shape().dim(0);
  assert(batch_size == _ctx.at(input_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_in_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_out_index).shape().dim(0));
  assert(_ctx.at(input_index).shape().dim(1) == _ctx.at(weights_index).shape().dim(1));

  const auto num_units = _ctx.at(output_index).shape().dim(1);
  assert(num_units == _ctx.at(weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(0) &&
         num_units == _ctx.at(bias_index).shape().dim(0));
  assert(num_units == _ctx.at(output_index).shape().dim(1) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_in_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_out_index).shape().dim(1));

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index, asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                                     _ctx.at(output_index).type()));
  _builder.addShapeConstr(hidden_state_out_index,
                          asTensorInfo(asTensorShape(_ctx.at(hidden_state_out_index).shape()),
                                       _ctx.at(hidden_state_out_index).type()));
  _builder.addShapeConstr(input_index, asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                                    _ctx.at(input_index).type()));
  _builder.addShapeConstr(weights_index, asTensorInfo(asTensorShape(_ctx.at(weights_index).shape()),
                                                      _ctx.at(weights_index).type()));
  _builder.addShapeConstr(recurrent_weights_index,
                          asTensorInfo(asTensorShape(_ctx.at(recurrent_weights_index).shape()),
                                       _ctx.at(recurrent_weights_index).type()));
  _builder.addShapeConstr(bias_index, asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                                   _ctx.at(bias_index).type()));
  _builder.addShapeConstr(hidden_state_in_index,
                          asTensorInfo(asTensorShape(_ctx.at(hidden_state_in_index).shape()),
                                       _ctx.at(hidden_state_in_index).type()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int hidden_state_out_index;

    int input_index;
    int weights_index;
    int recurrent_weights_index;
    int bias_index;
    int hidden_state_in_index;

    FuseCode activation;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.hidden_state_out_index = hidden_state_out_index.asInt();

  param.input_index = input_index.asInt();
  param.weights_index = weights_index.asInt();
  param.recurrent_weights_index = recurrent_weights_index.asInt();
  param.bias_index = bias_index.asInt();
  param.hidden_state_in_index = hidden_state_in_index.asInt();
  param.activation = static_cast<FuseCode>(_ctx.at(fused_activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto hidden_state_out_alloc =
        ctx.at(::internal::tflite::operand::Index{param.hidden_state_out_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto weights_alloc = ctx.at(::internal::tflite::operand::Index{param.weights_index});
    auto recurrent_weights_alloc =
        ctx.at(::internal::tflite::operand::Index{param.recurrent_weights_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});
    auto hidden_state_in_alloc =
        ctx.at(::internal::tflite::operand::Index{param.hidden_state_in_index});
    auto act_info = asActivationInfo(param.activation);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLCopy> copy_fn{new ::arm_compute::CLCopy};
      copy_fn->configure(CAST_CL(hidden_state_in_alloc), CAST_CL(hidden_state_out_alloc));
      builder.append("COPY", std::move(copy_fn));

      std::unique_ptr<::arm_compute::CLRNNLayer> rnn_fn{new ::arm_compute::CLRNNLayer};

      // The hidden_state_in's data must be copied to hidden_state_out_alloc before fn->run() is
      // performed.
      rnn_fn->configure(CAST_CL(input_alloc), CAST_CL(weights_alloc),
                        CAST_CL(recurrent_weights_alloc), CAST_CL(bias_alloc),
                        CAST_CL(hidden_state_out_alloc), CAST_CL(output_alloc), act_info);

      builder.append("RNN", std::move(rnn_fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LSTM::Node &node)
{
  // TODO Implement LSTM op
  throw std::runtime_error("Not supported, yet");
}

void Planner::visit(const ::internal::tflite::op::Transpose::Node &node)
{
  VERBOSE(Transpose) << "Configure Transpose operation" << std::endl;
  // Transpose supports only height-wight dimention support.
  // CLPermute can be used to implement generic transpose along any axis
  // But CLPermute only implements [2,0,1], [1,2,0], [3,2,0,1]

  // TODO Implement other permutation CLPermute function and provide generic transpose
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  // NNAPI spec provides permutation vector for generic transpose
  // TODO Make the permutation vector a part of Param
  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    const auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    // CLTranspose assumes only spatial transpose, will be replaced with CLPermute
    // TODO Check the validity of permutation vector, then call CLPermute with permu vector
    auto fn = nnfw::make_unique<::arm_compute::CLTranspose>();

    fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc));

    builder.append("Transpose", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Floor::Node &node)
{
  VERBOSE(Floor) << "Configure Floor operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().output_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().input_index};

  // Set shape constraints
  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLFloor>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc));

      builder.append("Floor", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NEFloor>();

      fn->configure(ifm_alloc, ofm_alloc);

      builder.append("Floor", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::RSQRT::Node &node)
{
  VERBOSE(RSQRT) << "Configure Rsqrt operation" << std::endl;

  throw std::runtime_error("Not supported, yet");
}

void Planner::visit(const ::internal::tflite::op::SquaredDifference::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<SquaredDifferenceOperation>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE, 1.0,
                    ::arm_compute::RoundingPolicy::TO_NEAREST_EVEN);

      builder.append("SquaredDifference", std::move(fn));
    }
    else // NEON
    {
      auto fn = nnfw::make_unique<SquaredDifferenceOperation>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE, 1.0,
                    ::arm_compute::RoundingPolicy::TO_ZERO);

      builder.append("SquaredDifference", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Split::Node &node)
{
  VERBOSE(Split) << "Configure Split operation" << std::endl;

  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  int32_t axis = _ctx.at(axis_index).asScalar<int32_t>();

  // Handle negative axis
  if (axis < 0)
  {
    axis += ifm_shape.rank();
  }

  const int32_t num_split = node.param().ofm_indexes.size();
  const auto input_size = ifm_shape.dim(axis);
  assert(input_size % num_split == 0);
  const int32_t slice_size = input_size / num_split;

  // Set Shape Constraints and TensorInfo (for input)
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Set Shape Constraints and TensorInfo (for output)
  const auto rank = ifm_shape.rank();
  const uint32_t coord_index = ToARMComputeAxis(rank, axis).value();
  uint32_t depth = 0;

  ::arm_compute::Coordinates coordinates;
  coordinates.set_num_dimensions(rank);

  for (const auto &index : node.param().ofm_indexes)
  {
    const ::internal::tflite::operand::Index ofm_index{index};

    coordinates[coord_index] = depth;

    _builder.addSubsumptionConstr(ofm_index, ifm_index, coordinates,
                                  asTensorShape(_ctx.at(ofm_index).shape()), true);
    depth += slice_size;
  }

  // NOTE Split has no actual operation!
}

void Planner::visit(const ::internal::tflite::op::Pad::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index paddings_index{node.param().paddings_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto paddings_shape = _ctx.at(paddings_index).shape().asTensor();

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      paddings_index,
      asTensorInfo(asTensorShape(_ctx.at(paddings_index).shape()), _ctx.at(paddings_index).type(),
                   _ctx.at(paddings_index).scale(), _ctx.at(paddings_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int32_t padding_size;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  assert(_ctx.at(paddings_index).hasData() == true);

  // TODO: Currently we are supporting uniform padding for the tensor, so only a single
  //      value is being read. (TOP = BOTTOM = LEFT = RIGHT).
  //      Need to read padding values for all the sides (TOP, BOTTOM, LEFT & RIGHT)

  const auto &padding_data = _ctx.at(paddings_index).data();
  auto base = padding_data.base();
  auto padsize = reinterpret_cast<const int *>(base) + 3;
  param.padding_size = *padsize;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    auto fn = nnfw::make_unique<PadLayer>();

    fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.padding_size);
    builder.append("Pad", std::move(fn));

  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::SpaceToDepth::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index block_size_index{node.param().block_size_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape(), false),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape(), false),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int32_t block_size;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.block_size = _ctx.at(block_size_index).asScalar<int32_t>();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto rank = 4;

    auto fn = nnfw::make_unique<SimpleSpaceToDepth>();

    fn->configure(input_alloc, output_alloc, param.block_size, getARMComputeAxises(rank));
    builder.append("SpaceToDepth", std::move(fn));

  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Normalization::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape();
  const auto ifm_shape = _ctx.at(ifm_index).shape();

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  // {CL|Neon}L2Normalization performs the reduction only along dimension 0
  // L2 Normalization always performs the reduction along the depth axis
  // Thus, we repurpose {CL|Neon}NormalizationLayers to act as depthwise L2 normalizations by
  // choosing normalization parameters as below

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int32_t radius;
    float alpha;
    float beta;
    float bias;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.radius = 2 * ifm_shape.dim(3) + 1; // normSize = depth * 2 + 1
  param.alpha = 1.0f;                      // In the implementation to make alpha_ become 1
  param.beta = 0.5f;                       // pow(reduction, -0.5) = 1 / sqrt(reduction)
  param.bias = 0.0f;                       // Don't offset the reduction.

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const auto norm_info =
        ::arm_compute::NormalizationLayerInfo(::arm_compute::NormType::CROSS_MAP, param.radius,
                                              param.alpha, param.beta, param.bias, false);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = nnfw::make_unique<::arm_compute::CLNormalizationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), norm_info);

      builder.append("L2Normalize", std::move(fn));
    }
    else
    {
      auto fn = nnfw::make_unique<::arm_compute::NENormalizationLayer>();

      fn->configure(CAST_NE(ifm_alloc), CAST_NE(ofm_alloc), norm_info);

      builder.append("L2Normalize", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Pool2D::Implicit::Node &node)

{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

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

  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::L2,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStringInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("L2Pool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("L2Pool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Pool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::L2,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStringInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("L2Pool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("L2Pool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::EmbeddingLookup::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index lookups_index{node.param().lookups_index};
  const ::internal::tflite::operand::Index values_index{node.param().values_index};

  const auto &output_obj = _ctx.at(output_index);
  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &values_obj = _ctx.at(values_index);

  // Verify operand here, not at SimpleEmbeddingLookup::configure() to avoid acl's modifying
  // TensorShape sometimes(Issue: https://github.sec.samsung.net/STAR/nnfw/issues/729)
  {
    assert(lookups_obj.type() == ANEURALNETWORKS_TENSOR_INT32);

    const auto &output_shape = output_obj.shape();
    const auto &lookups_shape = lookups_obj.shape();
    const auto &values_shape = values_obj.shape();

    assert(lookups_shape.rank() == 1);
    assert(values_shape.rank() >= 2);

    // output should be a n-D tensor with the same rank and shape as the values tensor, except for
    // the first dimension which has the same size as lookups' only dimension.
    assert(output_shape.rank() == values_shape.rank());
    assert(output_shape.dim(0) == lookups_shape.dim(0));
    for (size_t n = 1; n < output_shape.rank(); ++n)
    {
      assert(output_shape.dim(n) == values_shape.dim(n));
    }
  }

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(output_obj.shape(), false), output_obj.type(),
                                       output_obj.scale(), output_obj.zeroPoint()));
  _builder.addShapeConstr(lookups_index,
                          asTensorInfo(asTensorShape(lookups_obj.shape()), lookups_obj.type(),
                                       lookups_obj.scale(), lookups_obj.zeroPoint()));
  _builder.addShapeConstr(values_index,
                          asTensorInfo(asTensorShape(values_obj.shape(), false), values_obj.type(),
                                       values_obj.scale(), values_obj.zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int32_t output_index;
    int32_t lookups_index;
    int32_t values_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.lookups_index = lookups_index.asInt();
  param.values_index = values_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto lookups_alloc = ctx.at(::internal::tflite::operand::Index{param.lookups_index});
    auto values_alloc = ctx.at(::internal::tflite::operand::Index{param.values_index});

    auto fn = nnfw::make_unique<SimpleEmbeddingLookup>();

    fn->configure(lookups_alloc, values_alloc, output_alloc);

    builder.append("EmbeddingLookup", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::HashtableLookup::Node &node)
{
  // TODO Implement HashtableLookup
  throw std::runtime_error("Not supported");
}

class AllocationContext final : public IAllocationContext
{
public:
  AllocationContext(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  ::arm_compute::ITensor *at(const ::internal::tflite::operand::Index &ind) const override
  {
    return _plan.operands().at(ind).ptr();
  }

private:
  ::internal::arm_compute::Plan &_plan;
};

class ExecutionBuilder final : public IExecutionBuilder
{
public:
  ExecutionBuilder(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  void append(const std::string &name, std::unique_ptr<::arm_compute::IFunction> &&f) override
  {
    _plan.operations().append(std::move(f));
    _plan.operations().at(_plan.operations().size() - 1).name() = name;
  }

#ifdef TFLITE_PROFILING_ENABLED
public:
  int plan_op_size() const { return _plan.operations().size(); }
  void addOpIndexToSteps(int from, int to, int op_idx)
  {
    for (int i = from; i < to; ++i)
      _plan.operations().at(i).op_idx() = op_idx;
  }
#endif

private:
  ::internal::arm_compute::Plan &_plan;
};

class PlanBuilder final : public IPlanBuilder
{
public:
  PlanBuilder(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  void addShapeConstr(const ::internal::tflite::operand::Index &ind,
                      const ::arm_compute::TensorInfo &info) override;

public:
  void addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                            const ::internal::tflite::operand::Index &base,
                            const ::arm_compute::Coordinates &offset,
                            const ::arm_compute::TensorShape &shape, bool extend_parent) override;

public:
  void addInitializer(const ::internal::tflite::operand::Index &ind,
                      const Initializer &initializer) override;

public:
  void addStage(const Stage &stage) override;

public:
  void finalize(void) const;

private:
  ::internal::arm_compute::Plan &_plan;

private:
  struct Subsumption
  {
  public:
    Subsumption(const ::internal::tflite::operand::Index &base,
                const ::arm_compute::Coordinates &offset, const ::arm_compute::TensorShape &shape,
                bool extend_parent)
        : _base{base}, _offset{offset}, _shape{shape}, _extend_parent{extend_parent}
    {
      // DO NOTHING
    }

  public:
    const ::internal::tflite::operand::Index &base(void) const { return _base; }
    const ::arm_compute::Coordinates &offset(void) const { return _offset; }
    const ::arm_compute::TensorShape &shape(void) const { return _shape; }
    const bool extend_parent(void) const { return _extend_parent; }

  private:
    const ::internal::tflite::operand::Index _base;
    const ::arm_compute::Coordinates _offset;
    const ::arm_compute::TensorShape _shape;
    const bool _extend_parent;
  };

private:
  std::map<int, ::arm_compute::TensorInfo> _tensor_info_ctx;
  std::map<int, std::shared_ptr<Subsumption>> _subsumption_ctx;
  std::map<int, Initializer> _initializer_ctx;
  std::vector<Stage> _stages;
};

void PlanBuilder::addShapeConstr(const ::internal::tflite::operand::Index &ind,
                                 const ::arm_compute::TensorInfo &info)
{
  _tensor_info_ctx[ind.asInt()] = info;
}

void PlanBuilder::addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                                       const ::internal::tflite::operand::Index &base,
                                       const ::arm_compute::Coordinates &offset,
                                       const ::arm_compute::TensorShape &shape, bool extend_parent)
{
  _subsumption_ctx[ind.asInt()] = std::make_shared<Subsumption>(base, offset, shape, extend_parent);
}

void PlanBuilder::addInitializer(const ::internal::tflite::operand::Index &ind,
                                 const Initializer &initializer)
{
  _initializer_ctx[ind.asInt()] = initializer;
}

void PlanBuilder::addStage(const Stage &stage) { _stages.emplace_back(stage); }

#include <stack>

void PlanBuilder::finalize(void) const
{
  // ITensor objects to be initialized later
  std::vector<std::shared_ptr<::arm_compute::ITensor>> tensors;

  // Create Tensor & CLSubTensor
  auto isAllocated = [this](int ind) {
    const ::internal::tflite::operand::Index operand_index{ind};
    return _plan.operands().exist(operand_index);
  };

  auto setCLTensor = [&](int ind) {
    auto tensor = std::make_shared<::arm_compute::CLTensor>();

    tensor->allocator()->init(_tensor_info_ctx.at(ind));

    // NOTE Do NOT allocate here. allocate should be invoked after configure functions
    _plan.operands().set(::internal::tflite::operand::Index{ind}, tensor);
    tensors.emplace_back(tensor);
  };

  auto setCLSubTensor = [&](int curr) {
    const auto &sub_info = *(_subsumption_ctx.find(curr)->second);

    auto base_tensor = _plan.operands().at(sub_info.base()).ptr();

    assert(base_tensor != nullptr);

    auto curr_tensor = std::make_shared<::arm_compute::CLSubTensor>(
        CAST_CL(base_tensor), sub_info.shape(), sub_info.offset(), sub_info.extend_parent());

    _plan.operands().set(::internal::tflite::operand::Index{curr}, curr_tensor);
  };

  auto setNETensor = [&](int ind) {
    auto tensor = std::make_shared<::arm_compute::Tensor>();

    tensor->allocator()->init(_tensor_info_ctx.at(ind));

    // NOTE Do NOT allocate here. allocate should be invoked after configure functions
    _plan.operands().set(::internal::tflite::operand::Index{ind}, tensor);
    tensors.emplace_back(tensor);
  };

  auto setNESubTensor = [&](int curr) {
    const auto &sub_info = *(_subsumption_ctx.find(curr)->second);

    auto base_tensor = _plan.operands().at(sub_info.base()).ptr();

    assert(base_tensor != nullptr);

    auto curr_tensor = std::make_shared<::arm_compute::SubTensor>(base_tensor, sub_info.shape(),
                                                                  sub_info.offset());

    _plan.operands().set(::internal::tflite::operand::Index{curr}, curr_tensor);
  };

  for (auto it = _subsumption_ctx.begin(); it != _subsumption_ctx.end(); ++it)
  {
    std::stack<int> stack;

    stack.push(it->first);

    while (!stack.empty())
    {
      const auto curr = stack.top();

      if (isAllocated(curr))
      {
        // Skip if already allocated
        stack.pop();
        continue;
      }

      auto it_s = _subsumption_ctx.find(curr);

      if (it_s == _subsumption_ctx.end())
      {
        if (::internal::arm_compute::isGpuMode())
          setCLTensor(curr);
        else
          setNETensor(curr);
        stack.pop();
        continue;
      }

      const auto &sub_info = *(it_s->second);

      if (isAllocated(sub_info.base().asInt()))
      {
        if (::internal::arm_compute::isGpuMode())
          setCLSubTensor(curr);
        else
          setNESubTensor(curr);
        stack.pop();
      }
      else
      {
        // Allocate base tensor first
        stack.push(sub_info.base().asInt());
      }
    }
  }

  for (auto it = _tensor_info_ctx.begin(); it != _tensor_info_ctx.end(); ++it)
  {
    if (isAllocated(it->first))
    {
      // Skip if already allocated
      continue;
    }

    if (::internal::arm_compute::isGpuMode())
      setCLTensor(it->first);
    else
      setNETensor(it->first);
  }

  // Process Stage
  AllocationContext allocation_context{_plan};
  ExecutionBuilder execution_builder{_plan};

  for (int idx = 0; idx < _stages.size(); idx++)
  {
    const auto &stage = _stages[idx];
#ifdef TFLITE_PROFILING_ENABLED
    int from = execution_builder.plan_op_size();
#endif
    stage(allocation_context, execution_builder);
#ifdef TFLITE_PROFILING_ENABLED
    int to = execution_builder.plan_op_size();
    execution_builder.addOpIndexToSteps(from, to, idx);
#endif
  }

  // Allocate Tensor Memory
  for (const auto &tensor : tensors)
  {
    if (::internal::arm_compute::isGpuMode())
    {
      auto cl_tensor = CAST_CL(tensor.get());
      cl_tensor->allocator()->allocate();
    }
    else
    {
      auto ne_tensor = CAST_NE(tensor.get());
      ne_tensor->allocator()->allocate();
    }
  }

  // Fill weight/bias
  for (auto it = _initializer_ctx.begin(); it != _initializer_ctx.end(); ++it)
  {
    const ::internal::tflite::operand::Index operand_index{it->first};
    _plan.operands().at(operand_index).access(it->second);
  }

  // Initialize CLTensors that have data in their corresponding NNAPI operand but are not
  // initialized yet
  const auto &operands = _plan.model().operands();
  for (int idx = 0; idx < operands.size(); ++idx)
  {
    const ::internal::tflite::operand::Index operand_idx{idx};
    if (isAllocated(idx) && operands.at(operand_idx).hasData() &&
        _initializer_ctx.find(idx) == _initializer_ctx.end())
    {
      auto rank = operands.at(operand_idx).shape().rank();
      auto base = operands.at(operand_idx).data().base();
      auto type = operands.at(operand_idx).type();
      auto shape = operands.at(operand_idx).shape();

      switch (rank)
      {
        case 0: // scalar
        {
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initVectorTensor<float>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initVectorTensor<int32_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_UINT32:
            {
              auto initializer = std::bind(initVectorTensor<uint32_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initVectorTensor<uint8_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown scalar type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 1: // vector
        {
          auto size = shape.asVector();
          switch (type)
          {
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initVectorTensor<float>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initVectorTensor<int32_t>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initVectorTensor<uint8_t>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 2: // matrix
        {
          const auto matrix_shape = shape.asMatrix();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initMatrixTensor<float>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initMatrixTensor<int32_t>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initMatrixTensor<uint8_t>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 3: // 3D tensor
        {
          const auto tensor_shape = shape.asTensor();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initTensor3D<float>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initTensor3D<int32_t>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initTensor3D<uint8_t>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 4: // feature
        {
          const auto feature_shape = shape.asFeature();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initFeatureTensor<float>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer =
                  std::bind(initFeatureTensor<int32_t>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer =
                  std::bind(initFeatureTensor<uint8_t>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        default:
          throw std::runtime_error("Not supported, yet");
          break;
      }
    }
  }
}

//
// NNAPI Implementation
//
int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  if ((model == nullptr) || (compilation == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (!model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  std::shared_ptr<const internal::tflite::Model> internal;

  model->release(internal);

  ANeuralNetworksCompilation *compilation_ptr = new ANeuralNetworksCompilation(internal);
  if (compilation_ptr == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *compilation = compilation_ptr;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                             int32_t preference)
{
  if (compilation == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // NOTE Pure CL runimte currently ignores this API call
  // TODO Use preference
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  if (compilation == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (::internal::arm_compute::isGpuMode())
  {
    arm_compute::CLScheduler::get().default_init();
    arm_compute::CLKernelLibraryEx::get().init("./cl_kernels/", cl::Context::getDefault(),
                                               cl::Device::getDefault());
  }

  const auto &operands = compilation->plan().model().operands();
  const auto &operations = compilation->plan().model().operations();

  PlanBuilder plan_builder{compilation->plan()};

  for (uint32_t n = 0; n < operations.size(); ++n)
  {
    operations.at(n).accept(Planner{operands, plan_builder});
  }

  plan_builder.finalize();

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  delete compilation;
}
