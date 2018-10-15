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

#include "MaxPoolLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

#define MAXPOOLING_PARAMETERS                               \
  uint32_t height = getSizeOfDimension(_inputShape, 1);     \
  uint32_t width = getSizeOfDimension(_inputShape, 2);      \
  uint32_t outHeight = getSizeOfDimension(_outputShape, 1); \
  uint32_t outWidth = getSizeOfDimension(_outputShape, 2);  \
                                                            \
  uint32_t paddingHeight = (uint32_t)_paddingTop;           \
  uint32_t paddingWidth = (uint32_t)_paddingLeft;

MaxPoolLayer::MaxPoolLayer()
    : _inputData(nullptr), _outputData(nullptr), _inputShape(), _outputShape(), _paddingLeft(0),
      _paddingTop(0), _paddingRight(0), _paddingBottom(0), _strideWidth(0), _strideHeight(0),
      _kernelWidth(0), _kernelHeight(0), _activation(ANEURALNETWORKS_FUSED_NONE),
      _inputType(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

bool MaxPoolLayer::maxPoolFloat32()
{

  MAXPOOLING_PARAMETERS
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(_activation, &output_activation_min, &output_activation_max);

  ::tflite::optimized_ops::MaxPool(
      reinterpret_cast<const float *>(_inputData), convertShapeToDims(_inputShape), _strideWidth,
      _strideHeight, paddingWidth, paddingHeight, _kernelWidth, _kernelHeight,
      output_activation_min, output_activation_max, reinterpret_cast<float *>(_outputData),
      convertShapeToDims(_outputShape));
  return true;
}
bool MaxPoolLayer::maxPoolQuant8()
{

  MAXPOOLING_PARAMETERS
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeUint8(_activation, _outputShape, &output_activation_min,
                                &output_activation_max);

  ::tflite::optimized_ops::MaxPool(_inputData, convertShapeToDims(_inputShape), _strideWidth,
                                   _strideHeight, paddingWidth, paddingHeight, _kernelWidth,
                                   _kernelHeight, output_activation_min, output_activation_max,
                                   _outputData, convertShapeToDims(_outputShape));
  return true;
}

void MaxPoolLayer::configure(uint8_t *inputData, const Shape inputShape, const uint32_t paddingLeft,
                             const uint32_t paddingRight, const uint32_t paddingTop,
                             const uint32_t paddingBottom, const uint32_t strideWidth,
                             const uint32_t strideHeight, const uint32_t kernelWidth,
                             const uint32_t kernelHeight, const FuseCode activation,
                             uint8_t *outputData, const Shape outputShape)
{
  _inputData = inputData;

  _inputShape = inputShape;
  _inputType = inputShape.type;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _kernelWidth = kernelWidth;
  _kernelHeight = kernelHeight;
  _activation = activation;
  _outputData = outputData;
  _outputShape = outputShape;
}

void MaxPoolLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    maxPoolFloat32();
  }
  else if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error{"MaxPoolLayer : Not tested for TENSOR_QUANT8_ASYMM"};
    // maxPoolQuant8();
  }
}

#undef MAXPOOLING_PARAMETERS

} // namespace cpu
} // namespace kernel
} // namespace neurun
