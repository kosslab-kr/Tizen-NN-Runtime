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

#ifndef __NEURUN_BACKEND_ACL_CL_STAGE_GENERATOR_H__
#define __NEURUN_BACKEND_ACL_CL_STAGE_GENERATOR_H__

#include "backend/IStageGenerator.h"

#include "graph/operand/Set.h"
#include "backend/acl_cl/TensorBuilder.h"

namespace neurun
{
namespace backend
{
namespace acl_cl
{

class StageGenerator : public IStageGenerator
{
public:
  StageGenerator(const neurun::graph::operand::Set &ctx,
                 const std::shared_ptr<TensorBuilder> &tensor_builder);

  virtual std::shared_ptr<ITensorBuilder> tensor_builder() override { return _tensor_builder; }

  virtual Stage generate(const graph::operation::Conv2D::Implicit::Node &node) override;
  virtual Stage generate(const graph::operation::DepthwiseConv2D::Implicit::Node &node) override;
  virtual Stage generate(const graph::operation::MaxPool2D::Implicit::Node &node) override;
  virtual Stage generate(const graph::operation::AvgPool2D::Implicit::Node &node) override;
  virtual Stage generate(const graph::operation::Concat::Node &node) override;
  virtual Stage generate(const graph::operation::FullyConnected::Node &node) override;
  virtual Stage generate(const graph::operation::Reshape::Node &node) override;
  virtual Stage generate(const graph::operation::Softmax::Node &node) override;
  virtual Stage generate(const graph::operation::NOP::Node &node) override;
private:
  const neurun::graph::operand::Set &_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
};

} // namespace acl_cl
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_ACL_CL_STAGE_GENERATOR_H__
