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

#ifndef __INTERNAL_KERNELS_ACL_CL_TENSOR_CONVERT_FROM_COMMON_LAYER_H__
#define __INTERNAL_KERNELS_ACL_CL_TENSOR_CONVERT_FROM_COMMON_LAYER_H__

#include <NeuralNetworks.h>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/core/CL/ICLTensor.h>

#include "internal/Model.h"
#include "internal/common/Tensor.h"

namespace neurun
{
namespace kernel
{
namespace acl_cl
{

class TensorConvertFromCommonLayer : public ::arm_compute::IFunction
{
public:
  TensorConvertFromCommonLayer() {}

public:
  bool convert();

  void configure(::internal::common::Tensor *inputTensor, ::arm_compute::ICLTensor *outputTensor,
                 const ::neurun::graph::operand::Shape &tensorShape);

  void run();

private:
  ::internal::common::Tensor *_inputTensor;
  ::arm_compute::ICLTensor *_outputTensor;

  ::neurun::graph::operand::Shape _tensorShape{1};
};

} // namespace acl_cl
} // namespace kernel
} // namespace neurun

#endif // __INTERNAL_KERNELS_ACL_CL_TENSOR_CONVERT_FROM_COMMON_LAYER_H__

#endif
