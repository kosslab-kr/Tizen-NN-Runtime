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
#include <NeuralNetworksEx.h>

#include <cassert>
#include <stdexcept>
#include <new>

#include "nnfw/std/memory.h"

#include "graph/Graph.h"
#include "frontend/wrapper/model.h"
#include "frontend/wrapper/memory.h"
#include "graph/operation/AvgPool2D.h"
#include "graph/operation/Concat.h"
#include "graph/operation/Conv2D.h"
#include "graph/operation/FullyConnected.h"
#include "graph/operation/MaxPool2D.h"
#include "graph/operation/Reshape.h"
#include "graph/operation/Softmax.h"

int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  if (model == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  *model = new (std::nothrow) ANeuralNetworksModel{};
  if (*model == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel *model) { delete model; }

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                    const ANeuralNetworksOperandType *type)
{
  if ((model == nullptr) || (type == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  // scale and zeroPoint should be zero for scalars and non-fixed point tensors
  // Quantized:
  //  scale: a 32 bit floating point value greater than zero
  //  zeroPoint: a 32 bit integer, in range [0, 255]
  if (type->type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM)
  {
    if (!(type->scale > 0.0f))
    {
      return ANEURALNETWORKS_BAD_DATA;
    }

    if ((type->zeroPoint < 0) || (type->zeroPoint > 255))
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  else if ((type->scale != 0.0f) || (type->zeroPoint != 0))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  // dimensionCount should be zero for scalars
  if ((type->dimensionCount != 0) &&
      ((type->type == ANEURALNETWORKS_FLOAT32) || (type->type == ANEURALNETWORKS_INT32) ||
       (type->type == ANEURALNETWORKS_UINT32)))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  ::neurun::graph::operand::Shape shape(type->dimensionCount);
  ::neurun::graph::operand::TypeInfo typeInfo((OperandCode)(type->type), type->scale,
                                              type->zeroPoint);

  for (uint32_t axis = 0; axis < type->dimensionCount; ++axis)
  {
    shape.dim(axis) = type->dimensions[axis];
  }

  model->deref().addOperand(shape, typeInfo);

  // NOTE We do NOT allocate CLTensor here as we do not how to interpret this one.
  //      TensorFlow Lite may interpret a rank-4 tensor either as a feature map (with batch) or
  //      a convolution kernel.

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                         const void *buffer, size_t length)
{
  if ((model == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  // Negative index value is not allowed
  if (index < 0)
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  const neurun::graph::operand::Index ind{static_cast<uint32_t>(index)};

  if (!model->deref().operands().exist(ind))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  auto &obj = model->deref().operands().at(ind);
  if (obj.operandSize() != length)
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (!obj.setAsConstant())
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  using ::neurun::graph::operand::CachedData;

  model->deref().setOperandValue(
      ind, nnfw::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(buffer), length));

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                                   const ANeuralNetworksMemory *memory,
                                                   size_t offset, size_t length)
{
  if ((model == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  // Negative index value is not allowed
  if (index < 0)
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  const neurun::graph::operand::Index ind{static_cast<uint32_t>(index)};

  if (!model->deref().operands().exist(ind))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  auto &obj = model->deref().operands().at(ind);
  if ((obj.operandSize() != length) || (memory->size() < (offset + length)))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (!obj.setAsConstant())
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  using ::neurun::graph::operand::ExternalData;

  model->deref().setOperandValue(
      ind, nnfw::make_unique<ExternalData>(
               reinterpret_cast<const uint8_t *>(memory->base() + offset), length));

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t *inputs, uint32_t outputCount,
                                      const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  for (uint32_t i = 0; i < outputCount; i++)
  {
    const ::neurun::graph::operand::Index ind{outputs[i]};
    auto &obj = model->deref().operands().at(ind);

    if (!obj.setAsOperationOutput())
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  auto &graph = model->deref();

  auto node_param =
      neurun::graph::operation::Node::InitParam{inputCount, inputs, outputCount, outputs};

  switch (type)
  {
    case ANEURALNETWORKS_CONV_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using GraphNode = neurun::graph::operation::Conv2D::Implicit::Node;

        graph.addOperation(nnfw::make_unique<GraphNode>(node_param));
      }
      else
      {
        throw std::runtime_error{"Explicit padding in Conv2D is not supported, yet"};
      }

      break;
    }
    case ANEURALNETWORKS_MAX_POOL_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using GraphNode = neurun::graph::operation::MaxPool2D::Implicit::Node;

        graph.addOperation(nnfw::make_unique<GraphNode>(node_param));
      }
      else
      {
        throw std::runtime_error{"Explicit padding in MaxPool2D is not supported, yet"};
      }

      break;
    }
    case ANEURALNETWORKS_AVERAGE_POOL_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using GraphNode = neurun::graph::operation::AvgPool2D::Implicit::Node;

        graph.addOperation(nnfw::make_unique<GraphNode>(node_param));
      }
      else
      {
        throw std::runtime_error{"Explicit padding in AvgPool2D is not supported, yet"};
      }

      break;
    }
    case ANEURALNETWORKS_CONCATENATION:
    {
      using GraphNode = neurun::graph::operation::Concat::Node;

      graph.addOperation(nnfw::make_unique<GraphNode>(node_param));

      break;
    }
    case ANEURALNETWORKS_RESHAPE:
    {
      using GraphNode = neurun::graph::operation::Reshape::Node;

      graph.addOperation(nnfw::make_unique<GraphNode>(node_param));

      break;
    }
    case ANEURALNETWORKS_FULLY_CONNECTED:
    {
      using GraphNode = neurun::graph::operation::FullyConnected::Node;

      graph.addOperation(nnfw::make_unique<GraphNode>(node_param));

      break;
    }
    case ANEURALNETWORKS_SOFTMAX:
    {
      using GraphNode = neurun::graph::operation::Softmax::Node;

      graph.addOperation(nnfw::make_unique<GraphNode>(node_param));

      break;
    }
    default:
      throw std::runtime_error{"Not supported operation"};
  };

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                        ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  for (uint32_t i = 0; i < outputCount; i++)
  {
    const ::neurun::graph::operand::Index ind{outputs[i]};
    auto &obj = model->deref().operands().at(ind);

    if (!obj.setAsOperationOutput())
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  // Workaround: to avoid compile error by unused-parameter, use inputCount
  if (inputCount == 0)
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  switch (type)
  {
    default:
      throw std::runtime_error{"Not supported operation"};
  }
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                                  const uint32_t *inputs, uint32_t outputCount,
                                                  const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  // NOTE ::neurun::graph::operand::Index uses int as its underlying type as various NNAPI
  //      functions such as ANeuralNetworksModel_setOperandValue use int to represent operand index
  //
  //      ANeuralNetworksModel_identifyInputsAndOutputs, however, uses uint32_t to represent operand
  //      index.
  //
  //      Below, static_cast<int>(...) is introduced to eliminate compiler warning.
  for (uint32_t n = 0; n < inputCount; ++n)
  {
    const neurun::graph::operand::Index ind{static_cast<uint32_t>(inputs[n])};
    model->deref().addInput(ind);

    auto &obj = model->deref().operands().at(ind);
    if (!obj.setAsModelInput())
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  for (uint32_t n = 0; n < outputCount; ++n)
  {
    const neurun::graph::operand::Index ind{static_cast<uint32_t>(outputs[n])};
    model->deref().addOutput(ind);

    auto &obj = model->deref().operands().at(ind);
    // Model output cannot become model input
    if (obj.isModelInput())
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  if (model == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  return model->finish();
}
