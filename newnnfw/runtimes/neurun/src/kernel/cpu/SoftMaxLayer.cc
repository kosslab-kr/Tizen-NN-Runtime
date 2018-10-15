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

#include "SoftMaxLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

SoftMaxLayer::SoftMaxLayer()
    : _inputData(nullptr), _outputData(nullptr), _beta(0.0), _inputShape(), _outputShape(),
      _inputType(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

bool SoftMaxLayer::softmaxFloat32()
{
  ::tflite::Dims<4> dim;
  if (getNumberOfDimensions(_inputShape) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_inputShape, 0);
    uint32_t input_size = getNumberOfElements(_inputShape) / batch_size;
    Shape shapeIn4D;
    shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
    dim = convertShapeToDims(shapeIn4D);
  }
  else if (getNumberOfDimensions(_inputShape) == 4)
  {
    dim = convertShapeToDims(_inputShape);
  }
  else
  {
    std::cout << "only 2D and 4D tensors supported" << std::endl;
    return false;
  }
  ::tflite::optimized_ops::Softmax(reinterpret_cast<const float *>(_inputData), dim, _beta,
                                   reinterpret_cast<float *>(_outputData), dim);
  return true;
}

bool SoftMaxLayer::softmaxQuant8()
{
  ::tflite::Dims<4> dim;
  if (getNumberOfDimensions(_inputShape) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_inputShape, 0);
    uint32_t input_size = getNumberOfElements(_inputShape) / batch_size;
    Shape shapeIn4D;
    shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
    dim = convertShapeToDims(shapeIn4D);
  }
  else if (getNumberOfDimensions(_inputShape) == 4)
  {
    dim = convertShapeToDims(_inputShape);
  }
  else
  {
    std::cout << "only 2D and 4D tensors supported" << std::endl;
    return false;
  }
  if (_outputShape.offset != 0 || _outputShape.scale != 1.f / 256)
  {
    std::cout << "incorrect scale / offset for output" << std::endl;
    return false;
  }
  static const int32_t kScaledDiffIntegerBits = 5;
  const double input_beta_real_multiplier = std::min(
      1.0 * _beta * _inputShape.scale * (1 << (31 - kScaledDiffIntegerBits)), (1ll << 31) - 1.0);
  int32_t input_multiplier = 0;
  int32_t input_left_shift = 0;
  if (!QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier, &input_multiplier,
                                        &input_left_shift))
  {
    return false;
  }
  float diff_min = -1.0f * CalculateInputRadius(kScaledDiffIntegerBits, input_left_shift);
  ::tflite::optimized_ops::Softmax(_inputData, dim, input_multiplier, input_left_shift, diff_min,
                                   _outputData, dim);
  return true;
}

void SoftMaxLayer::configure(uint8_t *inputData, const Shape &inputShape, const float beta,
                             uint8_t *outputData, const Shape &outputShape)
{
  _inputData = inputData;
  _inputShape = inputShape;
  _inputType = inputShape.type;
  _outputData = outputData;
  _outputShape = outputShape;
  _beta = beta;
}

void SoftMaxLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    softmaxFloat32();
  }
  else if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error{"SoftMaxLayer : Not tested for TENSOR_QUANT8_ASYMM"};
    // softmaxQuant8();
  }
}

} // namespace cpu
} // namespace kernel
} // namespace neurun
