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

#ifndef __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__
#define __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__

#include <NeuralNetworks.h>

#include <iostream>
#include <limits>
#include <vector>

#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "graph/operand/Object.h"
#include "graph/operand/DataType.h"

using OperandType = neurun::graph::operand::DataType;

namespace neurun
{
namespace kernel
{
namespace cpu
{

struct Shape
{
  OperandType type;
  std::vector<uint32_t> dimensions;
  float scale;
  int32_t offset;
};

uint32_t getNumberOfDimensions(const Shape &shape);

uint32_t getNumberOfElements(const Shape &shape);

uint32_t getSizeOfDimension(const Shape &shape, uint32_t dimensionIdx);

inline ::tflite::Dims<4> convertShapeToDims(const Shape &shape)
{
  // nnAssert(shape.dimensions.size() <= 4);
  ::tflite::Dims<4> dims;
  // The dimensions are reversed in Dims<4>.
  for (int i = 0; i < 4; ++i)
  {
    int src = static_cast<int>(shape.dimensions.size()) - i - 1;
    if (src >= 0)
    {
      dims.sizes[i] = static_cast<int>(getSizeOfDimension(shape, src));
    }
    else
    {
      dims.sizes[i] = 1;
    }
  }
  dims.strides[0] = 1;
  for (int i = 1; i < 4; i++)
  {
    dims.strides[i] = dims.strides[i - 1] * dims.sizes[i - 1];
  }
  return dims;
}

__wur bool QuantizeMultiplierSmallerThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                            int32_t *right_shift);

__wur bool GetQuantizedConvolutionMultipler(const Shape &inputShape, const Shape &filterShape,
                                            const Shape &biasShape, const Shape &outputShape,
                                            float *multiplier);
__wur bool QuantizeMultiplierGreaterThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                            int *left_shift);

void CalculateActivationRangeFloat(int32_t activation, float *activation_min,
                                   float *activation_max);

void CalculateActivationRangeUint8(int32_t activation, const Shape &outputShape, int32_t *act_min,
                                   int32_t *act_max);

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);

Shape getShape(const ::neurun::graph::operand::Object &o);

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t> &dimensions);

} // namespace cpu
} // namespace kernel
} // namespace neurun

#endif // __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__
