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

#ifndef __NEURUN_CODEGEN_PLAN_H__
#define __NEURUN_CODEGEN_PLAN_H__

#include "graph/Graph.h"
#include "codegen/operand/Context.h"
#include "codegen/operation/Sequence.h"

namespace neurun
{
namespace codegen
{

class Plan
{
public:
  Plan(const std::shared_ptr<neurun::graph::Graph> &model) : _model(model)
  {
    // DO NOTHING
  }

public:
  neurun::graph::Graph &model(void) { return *_model; }
  const neurun::graph::Graph &model(void) const { return *_model; }

public:
  operand::Context &operands(void) { return _operands; }
  const operand::Context &operands(void) const { return _operands; }

public:
  operation::Sequence &operations(void) { return _ops; }
  const operation::Sequence &operations(void) const { return _ops; }

private:
  std::shared_ptr<neurun::graph::Graph> _model;
  operand::Context _operands;
  operation::Sequence _ops;
};

} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_PLAN_H__
