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

#ifndef __INTERNAL_ISTAGE_GENERATOR_H__
#define __INTERNAL_ISTAGE_GENERATOR_H__

#include <memory>
#include <functional>

#include <arm_compute/runtime/IFunction.h>

#include "backend/ITensorBuilder.h"
#include "graph/operation/Conv2D.h"
#include "graph/operation/MaxPool2D.h"
#include "graph/operation/AvgPool2D.h"
#include "graph/operation/Concat.h"
#include "graph/operation/FullyConnected.h"
#include "graph/operation/Reshape.h"
#include "graph/operation/Softmax.h"
#include "graph/operation/NOP.h"
#include "graph/operation/Add.h"

struct IExecutionBuilder
{
  virtual ~IExecutionBuilder() = default;

  virtual void append(std::unique_ptr<::arm_compute::IFunction> &&f) = 0;
};

using Stage = std::function<void(IExecutionBuilder &)>;

namespace neurun
{
namespace backend
{

struct IStageGenerator
{
  virtual ~IStageGenerator() = default;

  virtual std::shared_ptr<ITensorBuilder> tensor_builder() = 0;

  virtual Stage generate(const graph::operation::Conv2D::Implicit::Node &node) = 0;
  virtual Stage generate(const graph::operation::MaxPool2D::Implicit::Node &node) = 0;
  virtual Stage generate(const graph::operation::AvgPool2D::Implicit::Node &node) = 0;
  virtual Stage generate(const graph::operation::Concat::Node &node) = 0;
  virtual Stage generate(const graph::operation::FullyConnected::Node &node) = 0;
  virtual Stage generate(const graph::operation::Reshape::Node &node) = 0;
  virtual Stage generate(const graph::operation::Softmax::Node &node) = 0;
  virtual Stage generate(const graph::operation::NOP::Node &node) = 0;
  virtual Stage generate(const graph::operation::Add::Node &node) = 0;
};

} // namespace backend
} // namespace neurun

#endif // __INTERNAL_ISTAGE_GENERATOR_H__
