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

#ifndef __NEURUN_CODEGEN_PLAN_BUILDER_H__
#define __NEURUN_CODEGEN_PLAN_BUILDER_H__

#include "IPlanBuilder.h"
#include "codegen/Plan.h"
#include "backend/IStageGenerator.h"
#include "backend/ITensorBuilder.h"

namespace neurun
{
namespace codegen
{

class ExecutionBuilder final : public IExecutionBuilder
{
public:
  ExecutionBuilder(codegen::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  void append(std::unique_ptr<::arm_compute::IFunction> &&f) override
  {
    _plan.operations().append(std::move(f));
  }

private:
  codegen::Plan &_plan;
};

class PlanBuilder final : public IPlanBuilder
{
public:
  PlanBuilder(codegen::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  void addShapeConstr(const ::neurun::graph::operand::Index &ind,
                      const ::arm_compute::TensorInfo &info) override;

public:
  void addInitializer(const ::neurun::graph::operand::Index &ind,
                      const Initializer &initializer) override;

public:
  void addStage(const Stage &stage) override;

public:
  // TODO Remove the argument `tensor_builders`
  void finalize(const backend::TensorBuilderSet &tensor_builders);

public:
  const std::map<int, ::arm_compute::TensorInfo> &tensor_info_ctx() { return _tensor_info_ctx; }

private:
  codegen::Plan &_plan;

private:
  std::map<int, ::arm_compute::TensorInfo> _tensor_info_ctx;
  std::map<int, Initializer> _initializer_ctx;
  std::vector<Stage> _stages;
};

} // namepsace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_PLAN_BUILDER_H__
