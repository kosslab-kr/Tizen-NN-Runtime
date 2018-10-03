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

#ifndef __NNFW_UTIL_TENSOR_INDEX_H__
#define __NNFW_UTIL_TENSOR_INDEX_H__

#include <cstdint>
#include <cstddef>

#include <vector>
#include <initializer_list>

namespace nnfw
{
namespace util
{
namespace tensor
{

struct Index
{
public:
  Index(size_t rank) { _offsets.resize(rank); }

public:
  Index(std::initializer_list<int32_t> offsets) : _offsets{offsets}
  {
    // DO NOTHING
  }

public:
  size_t rank(void) const { return _offsets.size(); }

public:
  int32_t at(size_t n) const { return _offsets.at(n); }
  int32_t &at(size_t n) { return _offsets.at(n); }

private:
  std::vector<int32_t> _offsets;
};

// This is used to convert NNAPI tensor index to ARM tensor index or vice versa
inline static Index copy_reverse(const Index &origin)
{
  size_t rank = origin.rank();
  Index target(rank);
  for (int i = 0; i < rank; i++)
    target.at(i) = origin.at(rank-1 -i);
  return target;
}

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_INDEX_H__
