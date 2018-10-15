#ifndef __INTERNAL_OP_SQUAREDDIFFERENCE_H__
#define __INTERNAL_OP_SQUAREDDIFFERENCE_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace SquaredDifference
{

struct Param
{
  int32_t ofm_index;

  int32_t lhs_index;
  int32_t rhs_index;
  int32_t activation_index;

  Param() = default;
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

class Node final : public op::Node
{
public:
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  virtual ~Node() = default;

public:
  const Param &param(void) const { return _param; }

public:
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace SquaredDifference
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_SQUAREDDIFFERENCE_H__
