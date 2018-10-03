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

#include "GenericFullyConnectedLayer.h"
#include "internal/arm_compute.h"

#include <arm_compute/core/Helpers.h>

void GenericFullyConnectedLayer::configure(::arm_compute::ITensor *input,
                                           ::arm_compute::ITensor *weights,
                                           ::arm_compute::ITensor *biases,
                                           ::arm_compute::ITensor *output, bool needs_reshape,
                                           ::arm_compute::TensorShape reshape)
{
  _input = input;
  _weights = weights;
  _biases = biases;
  _output = output;
  _needs_reshape = needs_reshape;

  // TODO Too many duplicated code. Revise below code.
  if (::internal::arm_compute::isGpuMode())
  {
    if (_needs_reshape)
    {
      // reshape
      auto_init_if_empty(*_cl_buffer.info(), _input->info()->clone()->set_tensor_shape(reshape));
      _generic_reshape.configure(CAST_CL(_input), &_cl_buffer);

      _cl_fc.configure(&_cl_buffer, CAST_CL(_weights), CAST_CL(_biases), CAST_CL(_output));

      // NOTE _cl_buffer is inaccessible from outside, and thus it is safe to invoke allocate here.
      _cl_buffer.allocator()->allocate();
    }
    else
    {
      _cl_fc.configure(CAST_CL(_input), CAST_CL(_weights), CAST_CL(_biases), CAST_CL(_output));
    }
  }
  else
  {
    if (_needs_reshape)
    {
      // reshape
      auto_init_if_empty(*_neon_buffer.info(), _input->info()->clone()->set_tensor_shape(reshape));
      _generic_reshape.configure(CAST_NE(_input), &_neon_buffer);

      _neon_fc.configure(&_neon_buffer, CAST_NE(_weights), CAST_NE(_biases), CAST_NE(_output));

      // NOTE _neon_buffer is inaccessible from outside, and thus it is safe to invoke allocate
      // here.
      _neon_buffer.allocator()->allocate();
    }
    else
    {
      _neon_fc.configure(CAST_NE(_input), CAST_NE(_weights), CAST_NE(_biases), CAST_NE(_output));
    }
  }
}

void GenericFullyConnectedLayer::run(void)
{
  if (::internal::arm_compute::isGpuMode())
  {
    if (_needs_reshape)
      _generic_reshape.run();

    _cl_fc.run();
  }
  else
  {
    if (_needs_reshape)
      _generic_reshape.run();

    _neon_fc.run();
  }
}
