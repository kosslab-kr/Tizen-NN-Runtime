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

#ifndef __INTERNAL_ITENSOR_BUILDER_H__
#define __INTERNAL_ITENSOR_BUILDER_H__

#include <map>
#include <arm_compute/core/TensorInfo.h>

#include "graph/operand/Index.h"
#include "codegen/Plan.h"

namespace neurun
{
namespace backend
{

struct ITensorBuilder
{
  virtual ~ITensorBuilder(void) = default;
  virtual void mark(const ::neurun::graph::operand::Index &ind) = 0;
  // TODO Add an interface for adding subsumption info
  virtual void prepare(codegen::Plan &plan,
                       const std::map<int, ::arm_compute::TensorInfo> &tensor_info_ctx) = 0;
  virtual void allocate(void) = 0;
};

} // namespace backend
} // namespace neurun

#include <set>
#include <memory>

namespace neurun
{
namespace backend
{

using TensorBuilderSet = std::set<std::shared_ptr<backend::ITensorBuilder>>;

} // namespace backend
} // namespace neurun

#endif // __INTERNAL_ITENSOR_BUILDER_H__
