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

#include "IndexSet.h"

#include <algorithm>

namespace neurun
{
namespace graph
{
namespace operand
{

IndexSet::IndexSet(std::initializer_list<Index> list) : _set(list)
{
  // DO NOTHING
}

IndexSet::IndexSet(std::initializer_list<int32_t> list)
{
  for (auto val : list)
  {
    _set.emplace_back(static_cast<uint32_t>(val));
  }
}

IndexSet::IndexSet(std::initializer_list<uint32_t> list)
{
  for (auto val : list)
  {
    _set.emplace_back(val);
  }
}

bool IndexSet::contains(const Index &index) const
{
  return std::find(_set.begin(), _set.end(), index) != _set.end();
}

} // namespace operand
} // namespace graph
} // namespace neurun
