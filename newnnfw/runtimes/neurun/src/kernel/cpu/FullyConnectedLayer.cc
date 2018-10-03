/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "FullyConnectedLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "kernel/cpu/OperationUtils.h"

#include <mutex>

namespace neurun
{
namespace kernel
{
namespace cpu
{

FullyConnectedLayer::FullyConnectedLayer()
    : _inputData(nullptr), _weightsData(nullptr), _biasData(nullptr), _outputData(nullptr),
      _inputShape(), _weightsShape(), _biasShape(), _outputShape(),
      _activation(ANEURALNETWORKS_FUSED_NONE), _inputType(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

// executionMutex is used to protect concurrent access of non-threadsafe resources
// like gemmlowp::GemmContext.
// std::mutex is safe for pthreads on Android.
static std::mutex executionMutex;
bool FullyConnectedLayer::fullyConnectedFloat32()
{
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(_activation, &output_activation_min, &output_activation_max);
  // b/80425683, optimized implementation produces incorrect results when the
  // number of input elements is the squre of batch_size.
  uint32_t batch_size = getSizeOfDimension(_outputShape, 0);
  uint32_t input_n_elements = getNumberOfElements(_inputShape);
  if (batch_size * batch_size == input_n_elements)
  {
    ::tflite::reference_ops::FullyConnected(
        reinterpret_cast<const float *>(_inputData), convertShapeToDims(_inputShape),
        reinterpret_cast<const float *>(_weightsData), convertShapeToDims(_weightsShape),
        reinterpret_cast<const float *>(_biasData), convertShapeToDims(_biasShape),
        output_activation_min, output_activation_max, reinterpret_cast<float *>(_outputData),
        convertShapeToDims(_outputShape));
  }
  else
  {
    ::tflite::optimized_ops::FullyConnected(
        reinterpret_cast<const float *>(_inputData), convertShapeToDims(_inputShape),
        reinterpret_cast<const float *>(_weightsData), convertShapeToDims(_weightsShape),
        reinterpret_cast<const float *>(_biasData), convertShapeToDims(_biasShape),
        output_activation_min, output_activation_max, reinterpret_cast<float *>(_outputData),
        convertShapeToDims(_outputShape));
  }
  return true;
}

bool FullyConnectedLayer::fullyConnectedQuant8()
{
  int32_t inputOffset = -_inputShape.offset;
  int32_t weightsOffset = -_weightsShape.offset;
  int32_t outputOffset = _outputShape.offset;
  float real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  // Caution : 'Convolution' can make misleading. It seems it is just math term.
  if (!GetQuantizedConvolutionMultipler(_inputShape, _weightsShape, _biasShape, _outputShape,
                                        &real_multiplier) ||
      !QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier, &output_shift))
  {
    return false;
  }
  CalculateActivationRangeUint8(_activation, _outputShape, &output_activation_min,
                                &output_activation_max);
  static gemmlowp::GemmContext gemm_context;
  // Prevent concurrent executions that access gemm_context.
  std::unique_lock<std::mutex> lock(executionMutex);
  // Alow gemmlowp automatically decide how many threads to use.
  gemm_context.set_max_num_threads(0);
  ::tflite::optimized_ops::FullyConnected(
      _inputData, convertShapeToDims(_inputShape), inputOffset, _weightsData,
      convertShapeToDims(_weightsShape), weightsOffset,
      reinterpret_cast<const int32_t *>(_biasData), convertShapeToDims(_biasShape), outputOffset,
      output_multiplier, output_shift, output_activation_min, output_activation_max, _outputData,
      convertShapeToDims(_outputShape), &gemm_context);
  return true;
}

void FullyConnectedLayer::configure(uint8_t *inputData, const Shape inputShape,
                                    uint8_t *weightsData, const Shape weightsShape,
                                    uint8_t *biasData, const Shape biasShape, FuseCode activation,
                                    uint8_t *outputData, const Shape outputShape)
{
  _inputData = inputData;
  _inputShape = inputShape;
  _inputType = inputShape.type;
  _weightsData = weightsData;
  _weightsShape = weightsShape;
  _biasData = biasData;
  _biasShape = biasShape;
  _activation = activation;
  _outputData = outputData;
  _outputShape = outputShape;
}

void FullyConnectedLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    fullyConnectedFloat32();
  }
  else if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error{"FullyConnectedLayer : Not tested for TENSOR_QUANT8_ASYMM"};
    // fullyConnectedQuant8();
  }
}

} // namespace cpu
} // namespace kernel
} // namespace neurun
