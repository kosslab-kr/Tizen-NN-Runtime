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

#include "backend/acl_cl/TensorBuilder.h"

#include <cassert>

#include "operand/Object.h"

namespace neurun
{
namespace backend
{
namespace acl_cl
{

TensorBuilder::TensorBuilder()
{
  // DO NOTHING
}

void TensorBuilder::mark(const ::neurun::graph::operand::Index &ind)
{
  assert(_tensors.size() == 0);

  _inds.insert(ind);
}

void TensorBuilder::prepare(codegen::Plan &plan,
                            const std::map<int, ::arm_compute::TensorInfo> &tensor_info_ctx)
{
  assert(_tensors.size() == 0);

  // TODO Handle SubTensor(subsumption)
  //      Currently this TensorBuilder does not have subsumption info yet

  for (auto ind_int : _inds)
  {
    ::neurun::graph::operand::Index ind{ind_int};
    auto tensor = std::make_shared<::arm_compute::CLTensor>();
    tensor->allocator()->init(tensor_info_ctx.at(ind.asInt()));
    plan.operands().set(ind, std::make_shared<operand::Object>(tensor));
    _tensors[ind] = tensor;
  }
}

void TensorBuilder::allocate(void)
{
  assert(_inds.size() == _tensors.size());

  for (const auto &tensor_entry : _tensors)
  {
    auto tensor = tensor_entry.second;
    tensor->allocator()->allocate();
  }
}

std::shared_ptr<::arm_compute::CLTensor>
TensorBuilder::at(const ::neurun::graph::operand::Index &ind)
{
  return _tensors.at(ind);
}

} // namespace acl_cl
} // namespace backend
} // namespace neurun
