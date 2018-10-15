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

#include "kernel/cpu/OperationUtils.h"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace neurun
{
namespace kernel
{
namespace cpu
{

uint32_t getNumberOfDimensions(const Shape &shape) { return shape.dimensions.size(); }

uint32_t getNumberOfElements(const Shape &shape)
{
  uint32_t count = 1;
  for (size_t i = 0; i < shape.dimensions.size(); i++)
  {
    count *= shape.dimensions[i];
  }
  return count;
}

uint32_t getSizeOfDimension(const Shape &shape, uint32_t dimensionIdx)
{
  if (dimensionIdx >= shape.dimensions.size())
  {
    // TODO, log the error
    return 0;
  }
  return shape.dimensions[dimensionIdx];
}

bool QuantizeMultiplierSmallerThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                      int32_t *right_shift)
{
  assert(double_multiplier >= 0.);
  assert(double_multiplier < 1.);
  if (double_multiplier == 0.)
  {
    *quantized_multiplier = 0;
    *right_shift = 0;
    return true;
  }
  assert(double_multiplier > 0.);
  const double q = std::frexp(double_multiplier, right_shift);
  *right_shift *= -1;
  int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    --*right_shift;
  }
  assert(*right_shift >= 0);
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
  return true;
}

bool GetQuantizedConvolutionMultipler(const Shape &inputShape, const Shape &filterShape,
                                      const Shape &biasShape, const Shape &outputShape,
                                      float *multiplier)
{
  const float input_product_scale = inputShape.scale * filterShape.scale;
  const float bias_scale = biasShape.scale;
  const float output_scale = outputShape.scale;
  // The following conditions must be guaranteed by the training pipeline.
  assert(std::abs(input_product_scale - bias_scale) <=
         1e-6 * std::min(input_product_scale, bias_scale));
  assert(input_product_scale >= 0);
  assert(input_product_scale < output_scale);
  *multiplier = input_product_scale / output_scale;
  return true;
}

bool QuantizeMultiplierGreaterThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                      int *left_shift)
{
  assert(double_multiplier > 1.);
  const double q = std::frexp(double_multiplier, left_shift);
  int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    ++*left_shift;
  }
  assert(*left_shift >= 0);
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
  return true;
}

void CalculateActivationRangeFloat(int32_t activation, float *activation_min, float *activation_max)
{
  if (activation == ANEURALNETWORKS_FUSED_RELU)
  {
    *activation_min = 0.f;
    *activation_max = std::numeric_limits<float>::max();
  }
  else if (activation == ANEURALNETWORKS_FUSED_RELU6)
  {
    *activation_min = 0.f;
    *activation_max = 6.f;
  }
  else if (activation == ANEURALNETWORKS_FUSED_RELU1)
  {
    *activation_min = -1.f;
    *activation_max = 1.f;
  }
  else if (activation == ANEURALNETWORKS_FUSED_NONE)
  {
    *activation_min = std::numeric_limits<float>::lowest();
    *activation_max = std::numeric_limits<float>::max();
  }
  else
  {
    std::cout << "Unsupported fused activation function." << std::endl;
  }
}

void CalculateActivationRangeUint8(int32_t activation, const Shape &outputShape, int32_t *act_min,
                                   int32_t *act_max)
{
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  const auto scale = outputShape.scale;
  const auto zero_point = outputShape.offset;
  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
  };
  if (activation == ANEURALNETWORKS_FUSED_RELU)
  {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  }
  else if (activation == ANEURALNETWORKS_FUSED_RELU6)
  {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  }
  else if (activation == ANEURALNETWORKS_FUSED_RELU1)
  {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  }
  else if (activation == ANEURALNETWORKS_FUSED_NONE)
  {
    *act_min = qmin;
    *act_max = qmax;
  }
  else
  {
    std::cout << "Unsupported fused activation function." << std::endl;
  }
}

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift)
{
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                    (1ll << (31 - input_integer_bits)) / (1ll << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int32_t>(std::floor(max_input_rescaled));
}

Shape getShape(const ::neurun::graph::operand::Object &o)
{
  Shape shape;

  shape.type = static_cast<OperandType>(static_cast<int32_t>(o.typeInfo().type()));
  shape.dimensions = std::vector<uint32_t>(o.shape().dims().begin(), o.shape().dims().end());
  shape.scale = o.typeInfo().scale();
  // shape.offset = _offset;

  return shape;
}

size_t sizeOfData(OperandType type, const std::vector<uint32_t> &dimensions)
{
  size_t size = 4;

  switch (type)
  {
    case OperandType::SCALAR_FLOAT32:
    case OperandType::SCALAR_INT32:
    case OperandType::SCALAR_UINT32:
    case OperandType::TENSOR_FLOAT32:
    case OperandType::TENSOR_INT32:
      size = 4;
      break;
    case OperandType::TENSOR_QUANT8_ASYMM:
      size = 1;
      break;
    default:
      throw std::runtime_error("Not supported operand type.");
      break;
  }

  for (auto d : dimensions)
  {
    size *= d;
  }

  return size;
}

} // namespace cpu
} // namespace kernel
} // namespace neurun
