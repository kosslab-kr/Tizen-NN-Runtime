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

#include "ReshapeLayer.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

ReshapeLayer::ReshapeLayer()
    : _inputData(nullptr), _outputData(nullptr), _inputShape(), _outputShape()
{
  // DO NOTHING
}

bool ReshapeLayer::reshapeGeneric()
{
  size_t count = sizeOfData(_inputShape.type, _inputShape.dimensions);
  memcpy(reinterpret_cast<void *>(_outputData), reinterpret_cast<const void *>(_inputData), count);
  return true;
}

void ReshapeLayer::configure(uint8_t *inputData, const Shape &inputShape, uint8_t *outputData,
                             const Shape &outputShape)
{
  _inputData = inputData;
  _inputShape = inputShape;
  _outputData = outputData;
  _outputShape = outputShape;
}

void ReshapeLayer::run() { reshapeGeneric(); }

} // namespace cpu
} // namespace kernel
} // namespace neurun
