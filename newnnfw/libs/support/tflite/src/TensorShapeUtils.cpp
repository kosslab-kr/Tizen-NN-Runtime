#include "support/tflite/TensorShapeUtils.h"

namespace nnfw
{
namespace support
{
namespace tflite
{

nnfw::util::tensor::Shape broadcast(const nnfw::util::tensor::Shape &lhs_shape,
                                    const nnfw::util::tensor::Shape &rhs_shape)
{
  const uint32_t lhs_rank = lhs_shape.rank();
  const uint32_t rhs_rank = rhs_shape.rank();
  const uint32_t out_rank = std::max(lhs_rank, rhs_rank);

  // TODO Simplify implementation
  std::vector<int32_t> lhs_normalized_dims;
  std::vector<int32_t> rhs_normalized_dims;

  for (uint32_t n = 0; n < out_rank - lhs_rank; ++n)
  {
    lhs_normalized_dims.emplace_back(1);
  }
  for (uint32_t axis = 0; axis < lhs_rank; ++axis)
  {
    lhs_normalized_dims.emplace_back(lhs_shape.dim(axis));
  }

  for (uint32_t n = 0; n < out_rank - rhs_rank; ++n)
  {
    rhs_normalized_dims.emplace_back(1);
  }
  for (uint32_t axis = 0; axis < rhs_rank; ++axis)
  {
    rhs_normalized_dims.emplace_back(rhs_shape.dim(axis));
  }

  nnfw::util::tensor::Shape out_shape(out_rank);

  for (uint32_t axis = 0; axis < out_rank; ++axis)
  {
    out_shape.dim(axis) = std::max(lhs_normalized_dims.at(axis), rhs_normalized_dims.at(axis));
  }

  return out_shape;
}

} // namespace tflite
} // namespace support
} // namespace nnfw
