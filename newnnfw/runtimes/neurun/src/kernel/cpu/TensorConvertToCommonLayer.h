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

//
// THIS FILE IS UNUSED BUT LEFT FOR FUTURE REFERNCE
//

#if 0

#ifndef __NEURUN_KERNEL_CPU_TENSOR_CONVERT_TO_COMMON_LAYER_H__
#define __NEURUN_KERNEL_CPU_TENSOR_CONVERT_TO_COMMON_LAYER_H__

#include <NeuralNetworks.h>

#include <arm_compute/runtime/IFunction.h>

#include "internal/Model.h"
#include "internal/common/Tensor.h"
#include "internal/cpu.h"

namespace neurun
{
namespace kernel
{
namespace cpu
{

class TensorConvertToCommonLayer : public ::arm_compute::IFunction
{
public:
  TensorConvertToCommonLayer() {}

public:
  bool convert();

  void configure(::internal::cpu::Tensor *inputTensor, ::internal::common::Tensor *outputTensor,
                 const Shape &tensorShape);

  void run();

private:
  ::internal::cpu::Tensor *_inputTensor;
  ::internal::common::Tensor *_outputTensor;

  Shape _tensorShape{1};
};

} // namespace cpu
} // namespace kernel
} // namespace neurun

#endif // __NEURUN_KERNEL_CPU_TENSOR_CONVERT_TO_COMMON_LAYER_H__

#endif
