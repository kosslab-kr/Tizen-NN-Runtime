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

#ifndef __NNFW_UTIL_FEATURE_INDEX_H__
#define __NNFW_UTIL_FEATURE_INDEX_H__

#include <cstdint>

namespace nnfw
{
namespace util
{
namespace feature
{

class Index
{
public:
  Index() = default;

public:
  Index(int32_t ch, int32_t row, int32_t col) : _batch{1}, _ch{ch}, _row{row}, _col{col}
  {
    // DO NOTHING
  }
  Index(int32_t batch, int32_t ch, int32_t row, int32_t col) : _batch{batch}, _ch{ch}, _row{row}, _col{col}
  {
    // DO NOTHING
  }

public:
  int32_t batch(void) const { return _batch; }
  int32_t ch(void) const { return _ch; }
  int32_t row(void) const { return _row; }
  int32_t col(void) const { return _col; }

public:
  int32_t &batch(void) { return _batch; }
  int32_t &ch(void) { return _ch; }
  int32_t &row(void) { return _row; }
  int32_t &col(void) { return _col; }

private:
  int32_t _batch;
  int32_t _ch;
  int32_t _row;
  int32_t _col;
};

} // namespace feature
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_FEATURE_INDEX_H__
