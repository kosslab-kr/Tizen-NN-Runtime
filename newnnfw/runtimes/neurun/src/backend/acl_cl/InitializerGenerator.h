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

#ifndef __NEURUN_BACKEND_ACL_CL_INITIALIZER_GENERATOR_H__
#define __NEURUN_BACKEND_ACL_CL_INITIALIZER_GENERATOR_H__

#include "backend/IInitializerGenerator.h"

#include "graph/operand/Set.h"

namespace neurun
{
namespace backend
{
namespace acl_cl
{

class InitializerGenerator : public IInitializerGenerator
{
public:
  InitializerGenerator(const neurun::graph::operand::Set &ctx);

  Initializer generateWeight(const graph::operation::Conv2D::Implicit::Node &node) override;
  Initializer generateWeight(const graph::operation::FullyConnected::Node &node) override;

  Initializer generateBias(const graph::operation::Conv2D::Implicit::Node &node) override;
  Initializer generateBias(const graph::operation::FullyConnected::Node &node) override;

private:
  const neurun::graph::operand::Set &_ctx;
};

} // namespace acl_cl
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_ACL_CL_INITIALIZER_GENERATOR_H__
