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

#ifndef __NEURUN_GRAPH_GRAPH_H__
#define __NEURUN_GRAPH_GRAPH_H__

#include <functional>

#include "graph/operation/Node.h"
#include "graph/operation/Set.h"
#include "graph/operand/IndexSet.h"
#include "graph/operand/Set.h"

namespace neurun
{
namespace linear
{
class Linear;
} // namespace linear
} // namespace neurun

namespace neurun
{
namespace graph
{

class Graph
{
private:
  enum class Phase
  {
    BUILDING,
    MODEL,
    LOWERED,
    LINEARIZED // Everything is moved to Linear object so this Graph object is no longer effective
  };

public:
  template <bool is_const> class Iterator
  {
  public:
    using GraphRef = typename std::conditional<is_const, const Graph &, Graph &>::type;
    using NodeRef =
        typename std::conditional<is_const, const operation::Node &, operation::Node &>::type;
    using IterFn = std::function<void(NodeRef)>;

  public:
    virtual ~Iterator() = default;
    virtual void iterate(GraphRef graph, const IterFn &fn) const = 0;
  };

  template <bool is_const = false> class DefaultIterator final : public Iterator<is_const>
  {
  public:
    using GraphRef = typename Iterator<is_const>::GraphRef;
    using NodeRef = typename Iterator<is_const>::NodeRef;
    using IterFn = typename Iterator<is_const>::IterFn;

  public:
    void iterate(GraphRef graph, const IterFn &fn) const;
  };
  using DefaultConstIterator = DefaultIterator<true>;

  template <bool is_const = false> class PostDfsIterator final : public Iterator<is_const>
  {
  public:
    using GraphRef = typename Iterator<is_const>::GraphRef;
    using NodeRef = typename Iterator<is_const>::NodeRef;
    using IterFn = typename Iterator<is_const>::IterFn;

  public:
    void iterate(GraphRef graph, const IterFn &fn) const;
  };
  using PostDfsConstIterator = PostDfsIterator<true>;

public:
  Graph(void) = default;

  // Graph Building
public:
  operand::Index addOperand(const operand::Shape &shape, const operand::TypeInfo &type);
  operation::Index addOperation(std::unique_ptr<operation::Node> &&node);
  operation::Index insertOperation(const operand::Index &prev_operand_index,
                                   const operation::Index &next_operation_index,
                                   std::unique_ptr<operation::Node> &&node);
  void setOperandValue(const operand::Index &ind, std::unique_ptr<operand::Data> &&data);
  void addInput(const operand::Index &ind);
  void addOutput(const operand::Index &ind);
  void finishBuilding(void);
  void lower(void);
  std::unique_ptr<linear::Linear> linearize(void);
  bool isBuildingPhase(void) const { return _phase == Phase::BUILDING; }

private:
  void initializeUseDef();

  // Accessors
public:
  const operand::IndexSet &getInputs() const { return _inputs; }
  const operand::IndexSet &getOutputs() const { return _outputs; }
  const operand::Set &operands() const { return _operands; }
  operand::Set &operands() { return _operands; } // TODO Remove this non-const accessor
  const operation::Set &operations() const { return _operations; }

private:
  Phase _phase{Phase::BUILDING};
  operation::Set _operations;
  operand::Set _operands;
  operand::IndexSet _inputs;
  operand::IndexSet _outputs;
};

} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_GRAPH_H__
