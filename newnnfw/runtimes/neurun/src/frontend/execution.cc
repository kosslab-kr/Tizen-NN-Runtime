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

#include <new>

#include "frontend/wrapper/compilation.h"
#include "frontend/wrapper/execution.h"
#include "frontend/wrapper/event.h"

#include "graph/operand/Index.h"

//
// NNAPI Implementation
//
int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                    ANeuralNetworksExecution **execution)
{
  if ((compilation == nullptr) || (execution == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  std::shared_ptr<const neurun::codegen::Plan> plan;

  compilation->publish(plan);

  *execution = new (std::nothrow) ANeuralNetworksExecution{plan};
  if (*execution == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                      const ANeuralNetworksOperandType * /* type */,
                                      const void *buffer, size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();

  // TODO Check type conflicts

  // NOTE The current implemenation assumes that every input is a feature map.
  // TODO Remove this assumption
  neurun::graph::operand::IO::Index input_index{index};

  const auto operand_index = execution->plan().model().getInputs().at(input_index);

  if (operands.at(operand_index).shape().rank() == 2)
  {
    assert(operands.at(operand_index).shape().dim(0) == 1);

    const auto len = operands.at(operand_index).shape().dim(1);

    execution->source<neurun::exec::VectorSource>(
        index, len, reinterpret_cast<const uint8_t *>(buffer), length);
  }
  else if (operands.at(operand_index).shape().rank() == 4)
  {
    const auto &operand_shape = operands.at(operand_index).shape().asFeature();

    execution->source<neurun::exec::FeatureSource>(
        index, operand_shape, reinterpret_cast<const uint8_t *>(buffer), length);
  }
  else
  {
    throw std::runtime_error{"Not supported, yet"};
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                       const ANeuralNetworksOperandType * /* type */, void *buffer,
                                       size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();

  // TODO Check type conflicts

  // NOTE The current implemenation assumes that every output is a feature map.
  // TODO Remove this assumption
  neurun::graph::operand::IO::Index output_index{index};

  const auto operand_index = execution->plan().model().getOutputs().at(output_index);

  if (operands.at(operand_index).shape().rank() == 2)
  {
    assert(operands.at(operand_index).shape().dim(0) == 1);

    const auto len = operands.at(operand_index).shape().dim(1);

    execution->sink<neurun::exec::VectorSink>(index, len, reinterpret_cast<uint8_t *>(buffer),
                                              length);
  }
  else if (operands.at(operand_index).shape().rank() == 4)
  {
    const auto &operand_shape = operands.at(operand_index).shape().asFeature();

    execution->sink<neurun::exec::FeatureSink>(index, operand_shape,
                                               reinterpret_cast<uint8_t *>(buffer), length);
  }
  else
  {
    throw std::runtime_error{"Not supported, yet"};
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                          ANeuralNetworksEvent **event)
{
  if ((execution == nullptr) || (event == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // TODO: Handle event
  *event = new (std::nothrow) ANeuralNetworksEvent{};
  if (*event == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  const auto &plan = execution->plan();
  const auto &model = plan.model();

  // Set input(s)
  for (uint32_t n = 0; n < model.getInputs().size(); ++n)
  {
    auto setter = [&](::arm_compute::ITensor &tensor) { execution->source(n).push(tensor); };

    neurun::graph::operand::IO::Index input_index{n};

    ::neurun::graph::operand::Index index{model.getInputs().at(input_index)};
    auto objects = plan.operands().at(index);

    for (auto object : objects)
    {
      object->access(setter);
    }
  }

  const auto &operations = execution->plan().operations();

  for (uint32_t n = 0; n < operations.size(); ++n)
  {
    operations.at(n).run();
  }

  // Get output(s)
  for (uint32_t n = 0; n < model.getOutputs().size(); ++n)
  {
    auto getter = [&](::arm_compute::ITensor &tensor) { execution->sink(n).pull(tensor); };

    neurun::graph::operand::IO::Index output_index{n};

    ::neurun::graph::operand::Index index{model.getOutputs().at(output_index)};
    auto objects = plan.operands().at(index);

    for (auto object : objects)
    {
      object->access(getter);
    }
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution * /* execution */) {}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution *execution,
                                                int32_t /* index */,
                                                const ANeuralNetworksOperandType * /* type */,
                                                const ANeuralNetworksMemory *memory,
                                                size_t /* offset */, size_t /* length */)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // NYI
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution *execution,
                                                 int32_t /* index */,
                                                 const ANeuralNetworksOperandType * /* type */,
                                                 const ANeuralNetworksMemory *memory,
                                                 size_t /* offset */, size_t /* length */)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // NYI
  return ANEURALNETWORKS_NO_ERROR;
}
