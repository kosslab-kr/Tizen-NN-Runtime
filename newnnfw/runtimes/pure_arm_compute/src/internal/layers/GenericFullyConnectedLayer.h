/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __GENERIC_FULLY_CONNECTED_LAYER_H__
#define __GENERIC_FULLY_CONNECTED_LAYER_H__

#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include "internal/layers/GenericReshapeLayer.h"

class GenericFullyConnectedLayer : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ITensor *input, ::arm_compute::ITensor *weights,
                 ::arm_compute::ITensor *biases, ::arm_compute::ITensor *output, bool needs_reshape,
                 ::arm_compute::TensorShape reshape);

public:
  void run(void) override;

private:
  ::arm_compute::ITensor *_input;
  ::arm_compute::ITensor *_weights;
  ::arm_compute::ITensor *_biases;
  ::arm_compute::ITensor *_output;

  // buffer for reshaping input tensor
  ::arm_compute::CLTensor _cl_buffer;
  ::arm_compute::Tensor _neon_buffer;

private:
  ::arm_compute::CLFullyConnectedLayer _cl_fc;
  ::arm_compute::NEFullyConnectedLayer _neon_fc;
  GenericReshapeLayer _generic_reshape;
  bool _needs_reshape;
};

#endif // __GENERIC_FULLY_CONNECTED_LAYER_H__
