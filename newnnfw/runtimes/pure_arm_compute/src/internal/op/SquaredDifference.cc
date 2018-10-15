#include "internal/op/SquaredDifference.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace SquaredDifference
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace SquaredDifference
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace SquaredDifference
{
Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 3 && outputCount == 1);

  ofm_index = outputs[0];

  // Each input should be interpreted as follows:
  //
  //  0 -> LHS Tensor Index
  //  1 -> RHS Tensor Index
  //  2 -> Activation Index
  lhs_index = inputs[0];
  rhs_index = inputs[1];
  activation_index = inputs[2];
}

} // namespace SquaredDifference
} // namespace op
} // namespace tflite
} // namespace internal
