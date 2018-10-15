#include "internal/op/L2Normalization.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace L2Normalization
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace L2Normalization
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace L2Normalization
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 1 && outputCount == 1);

  ofm_index = outputs[0];

  ifm_index = inputs[0];
}

} // namespace L2Normalization
} // namespace op
} // namespace tflite
} // namespace internal
