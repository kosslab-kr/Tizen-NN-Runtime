#include "Permute.h"

#include <cassert>

#include "NodeVisitor.h"

namespace neurun
{
namespace graph
{
namespace operation
{
namespace Permute
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

Node::Node(const operand::Index &input, const operand::Index &output)
{
  setInputs({input});
  setOutputs({output});
}

void Node::setInputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 1);

  graph::operation::Node::setInputs(indexes);
}

void Node::setOutputs(const operand::IndexSet &indexes)
{
  assert(indexes.size() == 1);

  graph::operation::Node::setOutputs(indexes);
}

} // namespace Permute
} // namespace operation
} // namespace graph
} // namespace neurun
