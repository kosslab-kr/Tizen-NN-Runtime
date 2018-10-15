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

#ifndef __INTERNAL_KERNEL_ACL_CL_CONCAT_LAYER_H__
#define __INTERNAL_KERNEL_ACL_CL_CONCAT_LAYER_H__

#include <NeuralNetworks.h>

#include <arm_compute/core/CL/ICLTensor.h>
#include <arm_compute/runtime/IFunction.h>

#include "graph/operand/DataType.h"

using OperandType = neurun::graph::operand::DataType;

namespace neurun
{
namespace kernel
{
namespace acl_cl
{

//
// neurun::kernel::acl_cl::ConcatLayer
// A naive implementation of ConcatLayer for ACL
//

class ConcatLayer : public ::arm_compute::IFunction
{
public:
  ConcatLayer();

public:
  void configure(const std::vector<::arm_compute::ICLTensor *> &input_allocs,
                 int32_t axis /* NNAPI tensor axis from NHWC order */,
                 ::arm_compute::ICLTensor *output_alloc);

  void run();

private:
  bool concatenationFloat32();

private:
  std::vector<::arm_compute::ICLTensor *> _input_allocs;
  ::arm_compute::ICLTensor *_output_alloc;
  int32_t _axis;
  OperandType _input_type;
};

} // namespace acl_cl
} // namespace kernel
} // namespace neurun

#endif // __INTERNAL_KERNEL_ACL_CL_CONCAT_LAYER_H__
