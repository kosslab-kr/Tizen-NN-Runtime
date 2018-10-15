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

#include <NeuralNetworks.h>

#include <algorithm>

#include <arm_compute/core/CL/ICLTensor.h>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include "internal/Convert.h"
#include "backend/acl_cl/kernel/View.h"
#include "backend/acl_cl/TensorBuilder.h"
#include "internal/nnapi/kernel/Reader.h"
#include "internal/Padding.h"
#include "backend/IInitializerGenerator.h"
#include "backend/IStageGenerator.h"

#include "compilation.h"
#include "model.h"
#include "logging.h"

#include "graph/dumper/Dumper.h"
#include "codegen/IPlanBuilder.h"
#include "codegen/Planner.h"
#include "codegen/PlanBuilder.h"

#include "linear/Linear.h"

int ANeuralNetworksCompilation::finish()
{
  auto &plan = this->plan();
  const auto &operands = plan.model().operands();

  plan.model().lower();
  auto linear = plan.model().linearize();

  // Dump ops
  linear->accept(neurun::graph::dumper::Dumper{});

  neurun::codegen::PlanBuilder plan_builder{plan};

  auto tensor_builders = linear->markTensors();

  linear->accept(neurun::codegen::Planner{operands, plan_builder});

  // TODO Add optimization passes
  plan_builder.finalize(tensor_builders);

  return ANEURALNETWORKS_NO_ERROR;
}
