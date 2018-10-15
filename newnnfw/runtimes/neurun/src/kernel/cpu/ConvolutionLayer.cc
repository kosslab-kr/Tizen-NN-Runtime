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

#include "ConvolutionLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "kernel/cpu/OperationUtils.h"

#include <mutex>

namespace neurun
{
namespace kernel
{
namespace cpu
{

// If possible we will use this static buffer for the tensor.
static constexpr int kStaticBufferSize = 1605632;
static char static_scratch_buffer[kStaticBufferSize];
static std::mutex executionMutex;

#define ANDROID_NN_CONV_PARAMETERS(Type)                                      \
  uint32_t height = getSizeOfDimension(_inputShape, 1);                       \
  uint32_t width = getSizeOfDimension(_inputShape, 2);                        \
  uint32_t kernelHeight = getSizeOfDimension(_kernelShape, 1);                \
  uint32_t kernelWidth = getSizeOfDimension(_kernelShape, 2);                 \
  uint32_t outHeight = getSizeOfDimension(_outputShape, 1);                   \
  uint32_t outWidth = getSizeOfDimension(_outputShape, 2);                    \
  uint32_t inDepth = getSizeOfDimension(_inputShape, 3);                      \
                                                                              \
  uint32_t paddingHeight = (uint32_t)_paddingTop;                             \
  uint32_t paddingWidth = (uint32_t)_paddingLeft;                             \
                                                                              \
  ::tflite::Dims<4> im2colDim;                                                \
  im2colDim.sizes[3] = (int)getSizeOfDimension(_outputShape, 0);              \
  im2colDim.sizes[2] = (int)getSizeOfDimension(_outputShape, 1);              \
  im2colDim.sizes[1] = (int)getSizeOfDimension(_outputShape, 2);              \
  im2colDim.sizes[0] = (int)inDepth * kernelHeight * kernelWidth;             \
                                                                              \
  im2colDim.strides[0] = 1;                                                   \
  for (int i = 1; i < 4; i++)                                                 \
  {                                                                           \
    im2colDim.strides[i] = im2colDim.strides[i - 1] * im2colDim.sizes[i - 1]; \
  }                                                                           \
  Type *im2colData = nullptr;                                                 \
  uint64_t im2colByteSize = sizeof(Type);                                     \
  std::unique_ptr<Type[]> im2colGuard;                                        \
  for (int i = 0; i < 4; i++)                                                 \
  {                                                                           \
    im2colByteSize *= im2colDim.sizes[i];                                     \
  }                                                                           \
  /* http://b/77982879, tflite::optimized_ops::Conv uses int for offsets */   \
  if (im2colByteSize >= 0x7fffffff)                                           \
  {                                                                           \
    std::cout << "Conv size is too large, not enough memory" << std::endl;    \
    return false;                                                             \
  }                                                                           \
  if (im2colByteSize <= kStaticBufferSize)                                    \
  {                                                                           \
    im2colData = reinterpret_cast<Type *>(static_scratch_buffer);             \
  }                                                                           \
  else                                                                        \
  {                                                                           \
    im2colData = new (std::nothrow) Type[im2colByteSize / sizeof(Type)];      \
    if (im2colData == nullptr)                                                \
    {                                                                         \
      std::cout << "Conv size is too large, not enough memory" << std::endl;  \
      return false;                                                           \
    }                                                                         \
    im2colGuard.reset(im2colData);                                            \
  }

ConvolutionLayer::ConvolutionLayer()
    : _inputData(nullptr), _kernelData(nullptr), _outputData(nullptr), _biasData(nullptr),
      _inputShape(), _kernelShape(), _outputShape(), _biasShape(), _paddingLeft(0), _paddingTop(0),
      _paddingRight(0), _paddingBottom(0), _strideWidth(0), _strideHeight(0),
      _activation(ANEURALNETWORKS_FUSED_NONE), _inputType(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

bool ConvolutionLayer::convFloat32()
{
  ANDROID_NN_CONV_PARAMETERS(float)

  const ::tflite::Dims<4> &kernel_dim = convertShapeToDims(_kernelShape);
  const int kernel_width = ArraySize(kernel_dim, 1);
  const int kernel_height = ArraySize(kernel_dim, 2);
  const bool need_im2col =
      _strideWidth != 1 || _strideHeight != 1 || kernel_width != 1 || kernel_height != 1;

  float *im2colDataToPass = nullptr;
  if (need_im2col)
  {
    im2colDataToPass = im2colData;
  }

  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(_activation, &output_activation_min, &output_activation_max);
  int32_t dilationWidthFactor = 1, dilationHeightFactor = 1;
  ::tflite::optimized_ops::Conv(
      reinterpret_cast<const float *>(_inputData), convertShapeToDims(_inputShape),
      reinterpret_cast<const float *>(_kernelData), convertShapeToDims(_kernelShape),
      reinterpret_cast<const float *>(_biasData), convertShapeToDims(_biasShape), _strideWidth,
      _strideHeight, dilationWidthFactor, dilationHeightFactor, paddingWidth, paddingHeight,
      output_activation_min, output_activation_max, reinterpret_cast<float *>(_outputData),
      convertShapeToDims(_outputShape), im2colDataToPass, im2colDim);
  return true;
}

bool ConvolutionLayer::convQuant8()
{
  ANDROID_NN_CONV_PARAMETERS(uint8_t)
  int32_t inputOffset = -_inputShape.offset;
  int32_t kernelOffset = -_kernelShape.offset;
  int32_t outputOffset = _outputShape.offset;
  float real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  if (!GetQuantizedConvolutionMultipler(_inputShape, _kernelShape, _biasShape, _outputShape,
                                        &real_multiplier) ||
      !QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier, &output_shift))
  {
    return false;
  }
  CalculateActivationRangeUint8(_activation, _outputShape, &output_activation_min,
                                &output_activation_max);
  static gemmlowp::GemmContext gemm_context;
  // Prevent concurrent executions that may access the scratch buffer and
  // gemm_context.
  std::unique_lock<std::mutex> lock(executionMutex);
  // Alow gemmlowp automatically decide how many threads to use.
  gemm_context.set_max_num_threads(0);
  ::tflite::optimized_ops::Conv(
      _inputData, convertShapeToDims(_inputShape), inputOffset, _kernelData,
      convertShapeToDims(_kernelShape), kernelOffset, reinterpret_cast<const int32_t *>(_biasData),
      convertShapeToDims(_biasShape), _strideWidth, _strideHeight, paddingWidth, paddingHeight,
      outputOffset, output_multiplier, output_shift, output_activation_min, output_activation_max,
      _outputData, convertShapeToDims(_outputShape), im2colData, im2colDim, &gemm_context);
  return true;
}

void ConvolutionLayer::configure(uint8_t *inputData, const Shape inputShape, uint8_t *kernelData,
                                 const Shape kernelShape, uint8_t *biasData, const Shape biasShape,
                                 const uint32_t paddingLeft, const uint32_t paddingRight,
                                 const uint32_t paddingTop, const uint32_t paddingBottom,
                                 const uint32_t strideWidth, const uint32_t strideHeight,
                                 const FuseCode activation, uint8_t *outputData,
                                 const Shape outputShape)
{
  _inputData = inputData;
  _inputShape = inputShape;
  _inputType = inputShape.type;
  _kernelData = kernelData;
  _kernelShape = kernelShape;
  _biasData = biasData;
  _biasShape = biasShape;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _activation = activation;
  _outputData = outputData;
  _outputShape = outputShape;
}

void ConvolutionLayer::run()
{
  if (_inputType == OperandType::TENSOR_FLOAT32)
  {
    convFloat32();
  }
  else if (_inputType == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error{"ConvolutionLayer : Not tested for TENSOR_QUANT8_ASYMM"};
    // convQuant8();
  }
}

#undef ANDROID_NN_CONV_PARAMETERS

} // namespace cpu
} // namespace kernel
} // namespace neurun
