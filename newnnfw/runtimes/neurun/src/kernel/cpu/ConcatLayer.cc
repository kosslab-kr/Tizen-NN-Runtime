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

#include "ConcatLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

ConcatLayer::ConcatLayer()
    : _inputDataPtrs(), _outputData(nullptr), _axis(0), _inputShapes(), _outputShape(),
      _inputType(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

bool ConcatLayer::concatenationFloat32()
{
  int num_inputs = _inputShapes.size();
  std::vector<::tflite::Dims<4> *> inputDimsPtr(num_inputs);
  std::vector<::tflite::Dims<4>> inputDims(num_inputs);
  for (int i = 0; i < num_inputs; i++)
  {
    inputDims[i] = convertShapeToDims(_inputShapes[i]);
    inputDimsPtr[i] = &inputDims[i];
  }

  std::vector<const float *> inputFloatPtrs;

  for (auto ptr : _inputDataPtrs)
  {
    inputFloatPtrs.emplace_back(reinterpret_cast<const float *>(ptr));
  }

  ::tflite::optimized_ops::Concatenation<::tflite::FusedActivationFunctionType::kNone, float>(
      getNumberOfDimensions(_outputShape) - _axis - 1, inputFloatPtrs.data(), inputDimsPtr.data(),
      num_inputs, reinterpret_cast<float *>(_outputData), convertShapeToDims(_outputShape));
  return true;
}
bool ConcatLayer::concatenationQuant8()
{
  int num_inputs = _inputShapes.size();
  std::vector<::tflite::Dims<4> *> inputDimsPtr(num_inputs);
  std::vector<::tflite::Dims<4>> inputDims(num_inputs);
  for (int i = 0; i < num_inputs; i++)
  {
    inputDims[i] = convertShapeToDims(_inputShapes[i]);
    inputDimsPtr[i] = &inputDims[i];
  }
  ::tflite::optimized_ops::Concatenation<::tflite::FusedActivationFunctionType::kNone, uint8_t>(
      getNumberOfDimensions(_outputShape) - _axis - 1, _inputDataPtrs.data(), inputDimsPtr.data(),
      num_inputs, _outputData, convertShapeToDims(_outputShape));
  return true;
}

void ConcatLayer::configure(const std::vector<const uint8_t *> &inputDataPtrs,
                            const std::vector<Shape> &inputShapes, int32_t axis,
                            uint8_t *outputData, const Shape outputShape)
{
  _inputDataPtrs = inputDataPtrs;

  for (auto shape : inputShapes)
  {
    _inputShapes.emplace_back(shape);
    _inputType = shape.type;
  }

  _axis = axis;

  _outputData = outputData;
  _outputShape = outputShape;
}

void ConcatLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    concatenationFloat32();
  }
  else if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error{"ConcatLayer : Not tested for TENSOR_QUANT8_ASYMM"};
    // concatenationQuant8();
  }
}

} // namespace cpu
} // namespace kernel
} // namespace neurun
