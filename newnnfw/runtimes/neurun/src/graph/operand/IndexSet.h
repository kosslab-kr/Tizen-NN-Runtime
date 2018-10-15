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

#ifndef __NEURUN_GRAPH_OPERAND_INDEX_SET_H__
#define __NEURUN_GRAPH_OPERAND_INDEX_SET_H__

#include <initializer_list>
#include <vector>

#include "Index.h"

namespace neurun
{
namespace graph
{
namespace operand
{

class IndexSet
{
public:
  IndexSet(void) = default;
  IndexSet(std::initializer_list<Index> list);
  IndexSet(std::initializer_list<int32_t> list);
  IndexSet(std::initializer_list<uint32_t> list);

public:
  void append(const Index &index) { _set.emplace_back(index); }

public:
  uint32_t size() const { return static_cast<uint32_t>(_set.size()); }
  const Index &at(IO::Index set_index) const { return _set.at(set_index.asInt()); }
  const Index &at(uint32_t index) const { return _set.at(index); }
  bool contains(const Index &index) const;

public:
  std::vector<Index>::const_iterator begin(void) const { return _set.begin(); }
  std::vector<Index>::const_iterator end(void) const { return _set.end(); }

private:
  std::vector<Index> _set;
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_INDEX_SET_H__
