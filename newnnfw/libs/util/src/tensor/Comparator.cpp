#include "util/tensor/Comparator.h"
#include "util/tensor/Zipper.h"

#include "util/fp32.h"

namespace nnfw
{
namespace util
{
namespace tensor
{

std::vector<Diff<float>> Comparator::compare(const Shape &shape, const Reader<float> &expected,
                                             const Reader<float> &obtained,
                                             Observer *observer) const
{
  std::vector<Diff<float>> res;

  zip(shape, expected, obtained) <<
      [&](const Index &index, float expected_value, float obtained_value) {
        const auto relative_diff = nnfw::util::fp32::relative_diff(expected_value, obtained_value);

        if (!_compare_fn(expected_value, obtained_value))
        {
          res.emplace_back(index, expected_value, obtained_value);
        }

        // Update max_diff_index, if necessary
        if (observer != nullptr)
        {
          observer->notify(index, expected_value, obtained_value);
        }
      };

  return res;
}

} // namespace tensor
} // namespace util
} // namespace nnfw
