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

#ifndef __NEURUN_KERNEL_CPU_SOFTMAXLAYER_H__
#define __NEURUN_KERNEL_CPU_SOFTMAXLAYER_H__

#include <NeuralNetworks.h>

#include <arm_compute/runtime/IFunction.h>

#include "kernel/cpu/OperationUtils.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

class SoftMaxLayer : public ::arm_compute::IFunction
{
public:
  SoftMaxLayer();

public:
  bool softmaxFloat32();

  bool softmaxQuant8();

  void configure(uint8_t *inputData, const Shape &inputShape, const float beta, uint8_t *outputData,
                 const Shape &outputShape);

  void run();

private:
  uint8_t *_inputData;
  uint8_t *_outputData;

  float _beta;

  Shape _inputShape;
  Shape _outputShape;

  OperandType _inputType;
};

} // namespace cpu
} // namespace kernel
} // namespace neurun

#endif // __NEURUN_KERNEL_CPU_SOFTMAXLAYER_H__
