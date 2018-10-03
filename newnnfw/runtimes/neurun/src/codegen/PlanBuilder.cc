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

#include "PlanBuilder.h"

namespace neurun
{
namespace codegen
{

void PlanBuilder::addShapeConstr(const ::neurun::graph::operand::Index &ind,
                                 const ::arm_compute::TensorInfo &info)
{
  _tensor_info_ctx[ind.asInt()] = info;
}

void PlanBuilder::addInitializer(const ::neurun::graph::operand::Index &ind,
                                 const Initializer &initializer)
{
  _initializer_ctx[ind.asInt()] = initializer;
}

void PlanBuilder::addStage(const Stage &stage) { _stages.emplace_back(stage); }

void PlanBuilder::finalize(const backend::TensorBuilderSet &tensor_builders)
{
  // Prepare tensors
  for (auto &tensor_builder : tensor_builders)
  {
    tensor_builder->prepare(_plan, _tensor_info_ctx);
  }

  // Process Stage
  ExecutionBuilder execution_builder{_plan};

  for (const auto &stage : _stages)
  {
    stage(execution_builder);
  }

  // TODO Add code for CPU/ACL tensor allocation
  // Allocate Tensor Memory for cl_tensors
  for (auto &tensor_builder : tensor_builders)
  {
    tensor_builder->allocate();
  }

  // Fill weight/bias
  for (auto it = _initializer_ctx.begin(); it != _initializer_ctx.end(); ++it)
  {
    const ::neurun::graph::operand::Index operand_index{it->first};
    auto objects = _plan.operands().at(operand_index);

    for (auto object : objects)
    {
      object->access(it->second);
    }
  }
}

} // namepsace codegen
} // namespace neurun
