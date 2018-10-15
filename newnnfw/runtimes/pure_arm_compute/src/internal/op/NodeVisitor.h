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

#ifndef __INTERNAL_OP_NODE_VISITOR_H__
#define __INTERNAL_OP_NODE_VISITOR_H__

#include "internal/op/Add.h"
#include "internal/op/Sub.h"
#include "internal/op/Mul.h"
#include "internal/op/Div.h"
#include "internal/op/Conv2D.h"
#include "internal/op/DepthwiseConv2D.h"
#include "internal/op/Dequantize.h"
#include "internal/op/MaxPool2D.h"
#include "internal/op/AvgPool2D.h"
#include "internal/op/Concat.h"
#include "internal/op/Reshape.h"
#include "internal/op/ResizeBilinear.h"
#include "internal/op/StridedSlice.h"
#include "internal/op/FullyConnected.h"
#include "internal/op/Softmax.h"
#include "internal/op/ReduceMax.h"
#include "internal/op/Cast.h"
#include "internal/op/TopKV2.h"
#include "internal/op/Gather.h"
#include "internal/op/ReLU.h"
#include "internal/op/ReLU1.h"
#include "internal/op/ReLU6.h"
#include "internal/op/Tanh.h"
#include "internal/op/Squeeze.h"
#include "internal/op/Logistic.h"
#include "internal/op/Mean.h"
#include "internal/op/Rnn.h"
#include "internal/op/Transpose.h"
#include "internal/op/Lstm.h"
#include "internal/op/Floor.h"
#include "internal/op/Split.h"
#include "internal/op/RSQRT.h"
#include "internal/op/Pad.h"
#include "internal/op/SpaceToDepth.h"
#include "internal/op/L2Pool2D.h"
#include "internal/op/EmbeddingLookup.h"
#include "internal/op/HashtableLookup.h"
#include "internal/op/L2Normalization.h"
#include "internal/op/SquaredDifference.h"

namespace internal
{
namespace tflite
{
namespace op
{

struct NodeVisitor
{
  virtual ~NodeVisitor() = default;

  virtual void visit(const Add::Node &) = 0;
  virtual void visit(const Sub::Node &) = 0;
  virtual void visit(const Mul::Node &) = 0;
  virtual void visit(const Div::Node &) = 0;
  virtual void visit(const Conv2D::Implicit::Node &) = 0;
  virtual void visit(const Conv2D::Explicit::Node &) = 0;
  virtual void visit(const DepthwiseConv2D::Implicit::Node &) = 0;
  virtual void visit(const DepthwiseConv2D::Explicit::Node &) = 0;
  virtual void visit(const Dequantize::Node &) = 0;
  virtual void visit(const MaxPool2D::Implicit::Node &) = 0;
  virtual void visit(const MaxPool2D::Explicit::Node &) = 0;
  virtual void visit(const AvgPool2D::Implicit::Node &) = 0;
  virtual void visit(const AvgPool2D::Explicit::Node &) = 0;
  virtual void visit(const Concat::Node &) = 0;
  virtual void visit(const Reshape::Node &) = 0;
  virtual void visit(const ResizeBilinear::Node &) = 0;
  virtual void visit(const StridedSlice::Node &) = 0;
  virtual void visit(const FullyConnected::Node &) = 0;
  virtual void visit(const Softmax::Node &) = 0;
  virtual void visit(const ReduceMax::Node &) = 0;
  virtual void visit(const Cast::Node &) = 0;
  virtual void visit(const TopKV2::Node &) = 0;
  virtual void visit(const Gather::Node &) = 0;
  virtual void visit(const ReLU::Node &) = 0;
  virtual void visit(const ReLU1::Node &) = 0;
  virtual void visit(const ReLU6::Node &) = 0;
  virtual void visit(const Tanh::Node &) = 0;
  virtual void visit(const Squeeze::Node &) = 0;
  virtual void visit(const Logistic::Node &) = 0;
  virtual void visit(const Mean::Node &) = 0;
  virtual void visit(const RNN::Node &) = 0;
  virtual void visit(const Transpose::Node &) = 0;
  virtual void visit(const LSTM::Node &) = 0;
  virtual void visit(const Floor::Node &) = 0;
  virtual void visit(const Split::Node &) = 0;
  virtual void visit(const RSQRT::Node &) = 0;
  virtual void visit(const Pad::Node &) = 0;
  virtual void visit(const SpaceToDepth::Node &) = 0;
  virtual void visit(const L2Pool2D::Implicit::Node &) = 0;
  virtual void visit(const L2Pool2D::Explicit::Node &) = 0;
  virtual void visit(const EmbeddingLookup::Node &) = 0;
  virtual void visit(const HashtableLookup::Node &) = 0;
  virtual void visit(const L2Normalization::Node &) = 0;
  virtual void visit(const SquaredDifference::Node &) = 0;
};

} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_NODE_VISITOR_H__
