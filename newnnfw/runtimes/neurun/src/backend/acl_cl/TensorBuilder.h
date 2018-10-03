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

#ifndef __NEURUN_BACKEND_ACL_CL_TENSOR_BUILDER_H__
#define __NEURUN_BACKEND_ACL_CL_TENSOR_BUILDER_H__

#include "backend/ITensorBuilder.h"

#include <unordered_map>
#include <unordered_set>

#include <arm_compute/runtime/CL/CLTensor.h>

namespace neurun
{
namespace backend
{
namespace acl_cl
{

class Plan;

class TensorBuilder : public ITensorBuilder
{
public:
  TensorBuilder();

  virtual void mark(const ::neurun::graph::operand::Index &ind) override;
  virtual void prepare(codegen::Plan &plan,
                       const std::map<int, ::arm_compute::TensorInfo> &tensor_info_ctx) override;
  virtual void allocate(void) override;

  std::shared_ptr<::arm_compute::CLTensor> at(const ::neurun::graph::operand::Index &ind);

private:
  std::unordered_set<graph::operand::Index> _inds;
  std::unordered_map<graph::operand::Index, std::shared_ptr<::arm_compute::CLTensor>> _tensors;
};

} // namespace acl_cl
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_ACL_CL_TENSOR_BUILDER_H__
