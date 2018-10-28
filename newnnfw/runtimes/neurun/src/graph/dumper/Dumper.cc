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

#include "Dumper.h"

#include <string>

#include "logging.h"

namespace neurun
{
namespace graph
{
namespace dumper
{

using namespace neurun::graph::operation;

void Dumper::visit(const Conv2D::Implicit::Node &node)
{
  VERBOSE(LIR) << "* Conv2D(Implicit)" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ") Kernel("
               << node.getInputs().at(1).value() << ") Bias(" << node.getInputs().at(2).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const MaxPool2D::Implicit::Node &node)
{
  VERBOSE(LIR) << "* MaxPool2D(Implicit)" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const AvgPool2D::Implicit::Node &node)
{
  VERBOSE(LIR) << "* AvgPool2D(Implicit)" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const Concat::Node &node)
{
  VERBOSE(LIR) << "* Concat" << std::endl;
  std::string inputs;
  for (auto i : node.getInputs())
  {
    inputs += std::to_string(i.value()) + ",";
  }
  VERBOSE(LIR) << "  - Inputs : IFM(" << inputs << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const FullyConnected::Node &node)
{
  VERBOSE(LIR) << "* FullyConnected" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ") Weight("
               << node.getInputs().at(1).value() << ") Bias(" << node.getInputs().at(2).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const Reshape::Node &node)
{
  VERBOSE(LIR) << "* Reshape" << std::endl;
  // TODO The shape index should be "node.getInputs().at(1).value()" but not valid for now
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ") Shape("
               << "?"
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const Softmax::Node &node)
{
  VERBOSE(LIR) << "* Softmax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const NOP::Node &node)
{
  VERBOSE(LIR) << "* NOP" << std::endl;
  std::string inputs, outputs;
  for (auto i : node.getInputs())
  {
    inputs += std::to_string(i.value()) + ",";
  }
  VERBOSE(LIR) << "  - Inputs : IFM(" << inputs << ")" << std::endl;
  for (auto i : node.getOutputs())
  {
    outputs += std::to_string(i.value()) + ",";
  }
  VERBOSE(LIR) << "  - Outputs : OFM(" << outputs << ")" << std::endl;
}

void Dumper::visit(const Permute::Node &node)
{
  VERBOSE(LIR) << "* Permute" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(0).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void Dumper::visit(const Add::Node &node)
{
  VERBOSE(LIR) << "* Add" << std::endl;
  VERBOSE(LIR) << "  - Inputs : LHS(" << node.getInputs().at(0).value() << ") RHS(" << node.getInputs().at(1).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Outputs : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

} // namespace dumper
} // namespace graph
} // namespace neurun
