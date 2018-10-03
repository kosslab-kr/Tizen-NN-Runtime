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

#ifndef __NEURUN_CODEGEN_I_PLAN_BUILDER_H__
#define __NEURUN_CODEGEN_I_PLAN_BUILDER_H__

#include "arm_compute/core/TensorInfo.h"
#include "backend/IStageGenerator.h"
#include "backend/IInitializerGenerator.h"

namespace neurun
{
namespace codegen
{

struct IPlanBuilder
{
  virtual ~IPlanBuilder() = default;

  virtual void addShapeConstr(const ::neurun::graph::operand::Index &ind,
                              const ::arm_compute::TensorInfo &info) = 0;
  virtual void addInitializer(const ::neurun::graph::operand::Index &ind,
                              const Initializer &initializer) = 0;
  virtual void addStage(const Stage &) = 0;
};

} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_I_PLAN_BUILDER_H__
