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

#ifndef __INTERNAL_IINITIALIZER_GENERATOR_H__
#define __INTERNAL_IINITIALIZER_GENERATOR_H__

#include "arm_compute/core/ITensor.h"

#include "graph/operation/Conv2D.h"
#include "graph/operation/FullyConnected.h"

using Initializer = std::function<void(::arm_compute::ITensor &)>;

namespace neurun
{
namespace backend
{

struct IInitializerGenerator
{
  virtual ~IInitializerGenerator() = default;

  virtual Initializer generateWeight(const graph::operation::Conv2D::Implicit::Node &node) = 0;
  virtual Initializer generateWeight(const graph::operation::FullyConnected::Node &node) = 0;

  virtual Initializer generateBias(const graph::operation::Conv2D::Implicit::Node &node) = 0;
  virtual Initializer generateBias(const graph::operation::FullyConnected::Node &node) = 0;
};

} // namespace backend
} // namespace neurun

#endif // __INTERNAL_IINITIALIZER_GENERATOR_H__
