/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __INTERNAL_NNAPI_FEATURE_READER_H__
#define __INTERNAL_NNAPI_FEATURE_READER_H__

#include "internal/nnapi/feature/Utils.h"

#include "util/feature/Reader.h"

namespace internal
{
namespace nnapi
{
namespace feature
{

template <typename T> class Reader;

template <> class Reader<float> final : public nnfw::util::feature::Reader<float>
{
public:
  Reader(const ::nnfw::util::feature::Shape &shape, const uint8_t *ptr, size_t len)
      : _shape{shape}, _ptr{ptr}, _len{len}
  {
    // DO NOTHING
  }

public:
  const nnfw::util::feature::Shape &shape(void) const { return _shape; }

public:
  float at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = index_of(_shape, ch, row, col);

    const auto arr = reinterpret_cast<const float *>(_ptr);

    return arr[index];
  }
  float at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = index_of(_shape, batch, ch, row, col);

    const auto arr = reinterpret_cast<const float *>(_ptr);

    return arr[index];
  }

private:
  nnfw::util::feature::Shape _shape;

private:
  const uint8_t *_ptr;
  const size_t _len;
};

} // namespace feature
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_FEATURE_READER_H__
