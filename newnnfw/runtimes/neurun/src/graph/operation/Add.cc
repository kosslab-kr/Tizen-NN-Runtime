#include "Add.h"

#include <cassert>

#include "NodeVisitor.h"
#include "LowerInfo.h"

namespace neurun
{
namespace graph
{
namespace operation
{
namespace Add
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

Node::Node(const graph::operation::Node::InitParam &init_param)
{
  assert(init_param.input_count == 3);
  assert(init_param.output_count == 1);


  setInputs({init_param.inputs[0], init_param.inputs[1]});
  setOutputs({init_param.outputs[0]});

  _param.activation_index = init_param.inputs[2];
}

void Node::setInputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 2);

  graph::operation::Node::setInputs(indexes);
}

void Node::setOutputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 1);

  graph::operation::Node::setOutputs(indexes);
}

} // namespace Add
} // namespace operation
} // namespace graph
} // namespace neurun
