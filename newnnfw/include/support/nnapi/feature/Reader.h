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

#ifndef __NNFW_SUPPORT_NNAPI_FEATURE_READER_H__
#define __NNFW_SUPPORT_NNAPI_FEATURE_READER_H__

#include "support/nnapi/feature/Utils.h"

#include "util/feature/Shape.h"
#include "util/feature/Reader.h"

namespace nnfw
{
namespace support
{
namespace nnapi
{
namespace feature
{

template<typename T> class Reader : public nnfw::util::feature::Reader<T>
{
public:
  Reader(const nnfw::util::feature::Shape &shape, const T *base)
    : _shape{shape}, _base{base}
  {
    // DO NOTHING
  }

public:
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return *(_base + indexOf(_shape, ch, row, col));
  }

private:
  nnfw::util::feature::Shape _shape;
  const T *_base;
};

} // namespace feature
} // namespace nnapi
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_NNAPI_FEATURE_READER_H__
