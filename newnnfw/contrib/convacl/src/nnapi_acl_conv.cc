/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <NeuralNetworks.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <map>
#include <cassert>
#include <memory>
#include <boost/format.hpp>
// ACL Headers
#include <arm_compute/graph.h>

#include "util/environment.h"
#include "io_accessor.h"

//
// Asynchronous Event
//
struct ANeuralNetworksEvent
{
};

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event)
{
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event)
{
  delete event;
}

//
// Memory
//
struct ANeuralNetworksMemory
{
  // 1st approach - Store all the data inside ANeuralNetworksMemory object
  // 2nd approach - Store metadata only, and defer data loading as much as possible
};

int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset, ANeuralNetworksMemory** memory)
{
  *memory = new ANeuralNetworksMemory;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory)
{
  delete memory;
}

//
// Model
//
struct ANeuralNetworksModel
{
  // ANeuralNetworksModel should be a factory for Graph IR (a.k.a ISA Frontend)
  // TODO Record # of operands
  uint32_t numOperands;

  ANeuralNetworksModel() : numOperands(0)
  {
    // DO NOTHING
  }
};

int ANeuralNetworksModel_create(ANeuralNetworksModel** model)
{
  *model = new ANeuralNetworksModel;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model)
{
  delete model;
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model, const ANeuralNetworksOperandType *type)
{
  model->numOperands += 1;
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index, const void* buffer, size_t length)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index, const ANeuralNetworksMemory* memory, size_t offset, size_t length)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model, ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model)
{
  return ANEURALNETWORKS_NO_ERROR;
}

//
// Compilation
//
struct ANeuralNetworksCompilation
{
  // ANeuralNetworksCompilation should hold a compiled IR
};

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation)
{
  *compilation = new ANeuralNetworksCompilation;
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation)
{
  return ANEURALNETWORKS_NO_ERROR;
}

//
// Execution
//
struct ANeuralNetworksExecution
{
  // ANeuralNetworksExecution corresponds to NPU::Interp::Session

  arm_compute::graph::frontend::Stream graph{0, "ACL_CONV"};
};

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation, ANeuralNetworksExecution** execution)
{
  std::cout << __FUNCTION__ << " +++" << std::endl;
  *execution = new ANeuralNetworksExecution;

  using arm_compute::DataType;
  using arm_compute::graph::Target;
  using arm_compute::graph::TensorDescriptor;
  using arm_compute::TensorShape;
  using arm_compute::graph::frontend::InputLayer;
  using arm_compute::graph::frontend::OutputLayer;

  ANeuralNetworksExecution* execlocal = *execution;
  arm_compute::graph::frontend::Stream& graph = execlocal->graph;

  Target target_hint = nnfw::util::get_env_int("NNFW_ACL_USENEON")
                           ? Target::NEON : Target::CL;
  bool autoinc = nnfw::util::get_env_bool("NNFW_TEST_AUTOINC");

  graph << target_hint
        << InputLayer(TensorDescriptor(TensorShape(3U, 3U, 1U, 1U), DataType::F32),
                  std::unique_ptr<InputAccessor>(new InputAccessor(autoinc)))
        << arm_compute::graph::frontend::ConvolutionLayer(
              3U, 3U, 1U,
              std::unique_ptr<WeightAccessor>(new WeightAccessor(autoinc)),
              std::unique_ptr<BiasAccessor>(new BiasAccessor()),
              arm_compute::PadStrideInfo(1, 1, 0, 0))
        << OutputLayer(
              std::unique_ptr<OutputAccessor>(new OutputAccessor()));

  std::cout << __FUNCTION__ << " ---" << std::endl;
  return ANEURALNETWORKS_NO_ERROR;
}

// ANeuralNetworksExecution_setInput and ANeuralNetworksExecution_setOutput specify HOST buffer for input/output
int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length)
{
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event)
{
  std::cout << __FUNCTION__ << " +++" << std::endl;
  *event = new ANeuralNetworksEvent;

  // graph.run() fails with segment fail when only target_hint is added.
  // after fix adding 'Tensor' we may call graph.run()
  arm_compute::graph::frontend::Stream& graph = execution->graph;
  graph.run();

  std::cout << __FUNCTION__ << " ---" << std::endl;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution)
{
  delete execution;
}
