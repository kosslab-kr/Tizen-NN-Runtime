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

#include "backend/acl_cl/InitializerGenerator.h"

#include <arm_compute/core/Coordinates.h>

#include "backend/acl_cl/kernel/View.h"
#include "internal/nnapi/kernel/Reader.h"
#include "util/kernel/IndexIterator.h"

namespace neurun
{
namespace backend
{
namespace acl_cl
{

InitializerGenerator::InitializerGenerator(const neurun::graph::operand::Set &ctx) : _ctx(ctx)
{
  // DO NOTHING
}

Initializer
InitializerGenerator::generateWeight(const graph::operation::Conv2D::Implicit::Node &node)
{
  const ::neurun::graph::operand::Index ker_index{node.getInputs().at(1)};

  const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
  auto ker_base = _ctx.at(ker_index).data().base();
  auto ker_size = _ctx.at(ker_index).data().size();

  return [ker_shape, ker_base, ker_size](::arm_compute::ITensor &tensor) {
    const ::internal::nnapi::kernel::Reader<float> from{ker_shape, ker_base, ker_size};
    ::internal::arm_compute::kernel::View<float> into{&tensor};

    ::nnfw::util::kernel::iterate(ker_shape)
        << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(nth, ch, row, col);
             into.at(nth, ch, row, col) = value;
           };
  };
}

Initializer InitializerGenerator::generateWeight(const graph::operation::FullyConnected::Node &node)
{
  const ::neurun::graph::operand::Index weight_index{node.getInputs().at(1)};
  const ::neurun::graph::operand::Index input_index{node.getInputs().at(0)};

  const auto num_output = _ctx.at(weight_index).shape().dim(0);
  auto weight_base = _ctx.at(weight_index).data().base();
  auto weight_size = _ctx.at(weight_index).data().size();

  // NOTE We assume that input is a feature map
  // TODO Remove this restriction!
  const auto ifm_shape = _ctx.at(input_index).shape().asFeature();

  return [num_output, ifm_shape, weight_base, weight_size](::arm_compute::ITensor &tensor) {
    const ::nnfw::util::kernel::Shape ker_shape{num_output, ifm_shape.C, ifm_shape.H, ifm_shape.W};
    const ::internal::nnapi::kernel::Reader<float> from{ker_shape, weight_base, weight_size};

    ::nnfw::util::kernel::iterate(ker_shape)
        << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(nth, ch, row, col);

             uint32_t offset = 0;

             // ARM Compute Library uses 'NCHW' ordering
             offset += nth * ifm_shape.C * ifm_shape.H * ifm_shape.W;
             offset += ch * ifm_shape.H * ifm_shape.W;
             offset += row * ifm_shape.W;
             offset += col;

             const ::arm_compute::Coordinates coordinate{offset};

             auto into = reinterpret_cast<float *>(tensor.ptr_to_element(coordinate));

             *into = value;
           };
  };
}

Initializer InitializerGenerator::generateBias(const graph::operation::Conv2D::Implicit::Node &node)
{
  // TODO Refactor so we can reuse the common code

  const ::neurun::graph::operand::Index bias_index{node.getInputs().at(2)};

  auto bias_base = _ctx.at(bias_index).data().base();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  return [bias_base, bias_size](::arm_compute::ITensor &tensor) {
    for (int32_t n = 0; n < bias_size; ++n)
    {
      const ::arm_compute::Coordinates coordinate{n};

      float *into = reinterpret_cast<float *>(tensor.ptr_to_element(coordinate));

      const float *from = reinterpret_cast<const float *>(bias_base) + n;
      const auto value = *from;

      *into = value;
    }
  };
}

Initializer InitializerGenerator::generateBias(const graph::operation::FullyConnected::Node &node)
{
  const ::neurun::graph::operand::Index bias_index{node.getInputs().at(2)};

  auto bias_base = _ctx.at(bias_index).data().base();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  return [bias_base, bias_size](::arm_compute::ITensor &tensor) {
    for (int32_t n = 0; n < bias_size; ++n)
    {
      const ::arm_compute::Coordinates coordinate{n};

      float *into = reinterpret_cast<float *>(tensor.ptr_to_element(coordinate));

      const float *from = reinterpret_cast<const float *>(bias_base) + n;
      const auto value = *from;

      *into = value;
    }
  };
}

} // namespace acl_cl
} // namespace backend
} // namespace neurun
