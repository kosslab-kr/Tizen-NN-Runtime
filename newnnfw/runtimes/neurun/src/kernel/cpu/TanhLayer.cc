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

#include "TanhLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

TanhLayer::TanhLayer()
    : _inputData(nullptr), _outputData(nullptr), _inputShape(), _outputShape()
{
  // DO NOTHING
}

bool TanhLayer::tanhFloat32()
{
  ::tflite::optimized_ops::Tanh(
      reinterpret_cast<const float *>(_inputData), convertShapeToDims(_inputShape),
      reinterpret_cast<float *>(_outputData), convertShapeToDims(_outputShape));
  return true;
}

/*
bool TanhLayer::tanhQuant8()
{
	static constexpr int kInputIntegerBits = 4;
	const double input_real_multiplier = params.scale * static_cast<double>(1 << (31 - kInputIntegerBits));

  ::tflite::optimized_ops::MaxPool(_inputData, convertShapeToDims(_inputShape), _inputShape.offset,
			                             
                                   _outputData, convertShapeToDims(_outputShape));
  return true;
}
*/

void TanhLayer::configure(uint8_t *inputData, const Shape inputShape, 
                             uint8_t *outputData, const Shape outputShape)
{
  _inputData = inputData;
  _inputShape = inputShape;
  _outputData = outputData;
  _outputShape = outputShape;
}

void TanhLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    tanhFloat32();
  }
  else /*if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)*/
  {
    throw std::runtime_error{"TanhLayer : Not Builded Yet"/*Not tested for TENSOR_QUANT8_ASYMM"*/};
    // tanhQuant8();
  }
}

} // namespace cpu
} // namespace kernel
} // namespace neurun
