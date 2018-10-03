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

#ifndef __NEURUN_GRAPH_OPERATION_NODE_H__
#define __NEURUN_GRAPH_OPERATION_NODE_H__

#include <memory>

#include "graph/operand/IndexSet.h"

namespace neurun
{
namespace graph
{
namespace operation
{

class LowerInfo;
struct NodeVisitor;

class Node
{
public:
  struct InitParam
  {
    uint32_t input_count;
    const uint32_t *inputs;
    uint32_t output_count;
    const uint32_t *outputs;
  };

public:
  Node();
  virtual ~Node();

public:
  virtual void accept(NodeVisitor &&) const = 0;

public:
  virtual const operand::IndexSet &getInputs() const { return _inputs; }
  virtual const operand::IndexSet &getOutputs() const { return _outputs; }
  // It's for only input/output tensors but const data.
  virtual void setInputs(const operand::IndexSet &indexes) { _inputs = indexes; }
  virtual void setOutputs(const operand::IndexSet &indexes) { _outputs = indexes; }

public:
  void lower_info(std::unique_ptr<LowerInfo> &&lower_info);
  const LowerInfo *lower_info() const;

private:
  operand::IndexSet _inputs;
  operand::IndexSet _outputs;
  std::unique_ptr<LowerInfo> _lower_info;
};

} // namespace operation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_NODE_H__
