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

#ifndef __NEURUN_GRAPH_OPERATION_INDEX_LIST_H__
#define __NEURUN_GRAPH_OPERATION_INDEX_LIST_H__

#include <initializer_list>
#include <list>

#include "Index.h"

namespace neurun
{
namespace graph
{
namespace operation
{

class IndexList
{
public:
  IndexList(void) = default;
  IndexList(std::initializer_list<Index> list);

public:
  void append(const Index &index) { _list.push_back(index); }
  void remove(const Index &index) { _list.remove(index); }

public:
  uint32_t size() const { return static_cast<uint32_t>(_list.size()); }
  const std::list<Index> &list() const { return _list; }
  bool contains(const Index &index) const;

private:
  std::list<Index> _list;
};

} // namespace operation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_INDEX_LIST_H__
