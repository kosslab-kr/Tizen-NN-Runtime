/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_UTIL_FEATURE_TEXT_FORMATTER_H__
#define __NNFW_UTIL_FEATURE_TEXT_FORMATTER_H__

#include "util/feature/Shape.h"
#include "util/feature/Reader.h"

#include <ostream>
#include <iomanip>
#include <limits>

namespace nnfw
{
namespace util
{
namespace feature
{

template <typename T> class TextFormatter
{
public:
  TextFormatter(const Shape &shape, const Reader<T> &data) : _shape(shape), _data(data)
  {
    // DO NOTHING
  }

public:
  const Shape &shape(void) const { return _shape; }
  const Reader<T> &data(void) const { return _data; }

private:
  const Shape &_shape;
  const Reader<T> &_data;
};

template <typename T> std::ostream &operator<<(std::ostream &os, const TextFormatter<T> &fmt)
{
  const auto &shape = fmt.shape();

  for (uint32_t ch = 0; ch < shape.C; ++ch)
  {
    os << "  Channel " << ch << ":" << std::endl;
    for (uint32_t row = 0; row < shape.H; ++row)
    {
      os << "    ";
      for (uint32_t col = 0; col < shape.W; ++col)
      {
        const auto value = fmt.data().at(ch, row, col);
        os << std::right;
        os << std::fixed;
        os << std::setw(std::numeric_limits<T>::digits10 + 2);
        os << std::setprecision(5);
        os << value;
        os << " ";
      }
      os << std::endl;
    }
  }

  return os;
}

} // namespace feature
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_TEXT_FORMATTER_H__
