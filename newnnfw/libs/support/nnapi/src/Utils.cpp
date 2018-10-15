#include "support/nnapi/Utils.h"

#include <cassert>

namespace nnfw
{
namespace support
{
namespace nnapi
{

const char *to_string(const PaddingCode &code)
{
  assert((ANEURALNETWORKS_PADDING_SAME == code) || (ANEURALNETWORKS_PADDING_VALID == code));

  switch (code)
  {
    case ANEURALNETWORKS_PADDING_SAME:
      return "ANEURALNETWORKS_PADDING_SAME";
    case ANEURALNETWORKS_PADDING_VALID:
      return "ANEURALNETWORKS_PADDING_VALID";
  }

  return nullptr;
}

} // namespace nnapi
} // namespace support
} // namespace nnfw
