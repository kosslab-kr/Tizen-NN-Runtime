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

#include "InitializerGenerator.h"

#include "internal/nnapi/kernel/Reader.h"
#include "internal/nnapi/kernel/View.h"
#include "util/kernel/IndexIterator.h"

#include "NeuralNetworks.h"

namespace neurun
{
namespace backend
{
namespace cpu
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
    ::internal::nnapi::kernel::View<float> into{&tensor};

    ::nnfw::util::kernel::iterate(ker_shape)
        << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(nth, ch, row, col);
             into.at(nth, row, col, ch) = value;
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
  auto weight_type = _ctx.at(weight_index).typeInfo().type();

  // NOTE We assume that input is a feature map
  // TODO Remove this restriction!
  const auto ifm_shape = _ctx.at(input_index).shape().asFeature();

  switch (weight_type)
  {
    case ::neurun::graph::operand::DataType::TENSOR_FLOAT32:
    {
      return [num_output, ifm_shape, weight_base, weight_size](::arm_compute::ITensor &tensor) {
        const ::nnfw::util::kernel::Shape ker_shape{num_output, ifm_shape.C, ifm_shape.H,
                                                    ifm_shape.W};
        const ::internal::nnapi::kernel::Reader<float> from{ker_shape, weight_base, weight_size};

        ::nnfw::util::kernel::iterate(ker_shape)
            << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
                 const auto value = from.at(nth, ch, row, col);

                 uint32_t offset = 0;

                 // NNAPI uses NHWC ordering
                 offset += nth * ifm_shape.H * ifm_shape.W * ifm_shape.C;
                 offset += row * ifm_shape.W * ifm_shape.C;
                 offset += col * ifm_shape.C;
                 offset += ch;

                 const ::arm_compute::Coordinates coordinate{offset};

                 auto into = reinterpret_cast<float *>(tensor.ptr_to_element(coordinate));

                 *into = value;
               };
      };
    }
    case ::neurun::graph::operand::DataType::TENSOR_QUANT8_ASYMM:
    {
      return [num_output, ifm_shape, weight_base, weight_size](::arm_compute::ITensor &tensor) {
        const ::nnfw::util::kernel::Shape ker_shape{num_output, ifm_shape.C, ifm_shape.H,
                                                    ifm_shape.W};
        const ::internal::nnapi::kernel::Reader<uint8_t> from{ker_shape, weight_base, weight_size};
        ::nnfw::util::kernel::iterate(ker_shape)
            << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
                 const auto value = from.at(nth, ch, row, col);
                 uint32_t offset = 0;

                 // NNAPI uses NHWC ordering
                 offset += nth * ifm_shape.H * ifm_shape.W * ifm_shape.C;
                 offset += row * ifm_shape.W * ifm_shape.C;
                 offset += col * ifm_shape.C;
                 offset += ch;

                 const ::arm_compute::Coordinates coordinate{offset};

                 auto into = reinterpret_cast<uint8_t *>(tensor.ptr_to_element(coordinate));

                 *into = value;
               };
      };
    }
    default:
    {
      throw std::runtime_error("Not supported weight type");
    }
  }
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
  auto bias_type = _ctx.at(bias_index).typeInfo().type();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  switch (bias_type)
  {
    case ::neurun::graph::operand::DataType::TENSOR_FLOAT32:
    {
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
    case ::neurun::graph::operand::DataType::TENSOR_QUANT8_ASYMM:
    {
      return [bias_base, bias_size](::arm_compute::ITensor &tensor) {
        for (int32_t n = 0; n < bias_size; ++n)
        {
          const ::arm_compute::Coordinates coordinate{n};

          uint8_t *into = reinterpret_cast<uint8_t *>(tensor.ptr_to_element(coordinate));

          const uint8_t *from = reinterpret_cast<const uint8_t *>(bias_base) + n;
          const auto value = *from;

          *into = value;
        }
      };
    }
    default:
    {
      throw std::runtime_error("Not supported bias type");
    }
  }
}

} // namespace cpu
} // namespace backend
} // namespace neurun
