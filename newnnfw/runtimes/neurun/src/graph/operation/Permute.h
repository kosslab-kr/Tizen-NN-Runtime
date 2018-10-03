#ifndef __NEURUN_GRAPH_OPERATION_PERMUTE_PERMUTE_H__
#define __NEURUN_GRAPH_OPERATION_PERMUTE_PERMUTE_H__

#include "graph/operation/Node.h"

namespace neurun
{
namespace graph
{
namespace operation
{
namespace Permute
{

class Node : public graph::operation::Node
{
public:
  virtual void accept(NodeVisitor &&) const override;

public:
  Node(const operand::Index &input, const operand::Index &output);

public:
  virtual void setInputs(const operand::IndexSet &indexes) override;
  virtual void setOutputs(const operand::IndexSet &indexes) override;
};

} // namespace Permute
} // namespace operation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_PERMUTE_PERMUTE_H__
