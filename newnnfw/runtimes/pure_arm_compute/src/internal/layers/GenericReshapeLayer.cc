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

#include "GenericReshapeLayer.h"
#include "internal/arm_compute.h"

void GenericReshapeLayer::configure(::arm_compute::ITensor *input, ::arm_compute::ITensor *output)
{
  _input = input;
  _output = output;

  // NOTE This vector comes from CLPermuteKernel implementation
  //
  // This implementation permutes a tensor of shape W / H / C into another tensor of shape C / W / H
  //
  //     Original | Permuted
  // 0 | W        | C (from 2)
  // 1 | H        | W (from 0)
  // 2 | C        | H (from 1)
  //
  const ::arm_compute::PermutationVector pv{2, 0, 1};

  if (::internal::arm_compute::isGpuMode())
  {
    _cl_permute.configure(CAST_CL(input), &_cl_permuted, pv);
    _cl_reshape.configure(&_cl_permuted, CAST_CL(output));

    // NOTE _permuted is inaccessible from outside, and thus it is safe to invoke allocate here.
    _cl_permuted.allocator()->allocate();
  }
  else
  {
    _neon_permute.configure(CAST_NE(input), &_neon_permuted, pv);
    _neon_reshape.configure(&_neon_permuted, CAST_NE(output));

    // NOTE _permuted is inaccessible from outside, and thus it is safe to invoke allocate here.
    _neon_permuted.allocator()->allocate();
  }
}

void GenericReshapeLayer::run(void)
{
  if (::internal::arm_compute::isGpuMode())
  {
    _cl_permute.run();
    _cl_reshape.run();
  }
  else
  {
    _neon_permute.run();
    _neon_reshape.run();
  }
}
