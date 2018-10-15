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

#include "Set.h"

#include <cassert>

namespace neurun
{
namespace graph
{
namespace operation
{

const Index Set::generateIndex()
{
  assert((_index_count) <= 0x7fffffff);

  return Index{_index_count++};
}

Index Set::append(std::unique_ptr<Node> &&node)
{
  auto index = generateIndex();

  _nodes[index] = std::move(node);
  return index;
}

const Node &Set::at(const Index &index) const { return *(_nodes.at(index)); }

Node &Set::at(const Index &index) { return *(_nodes.at(index)); }

bool Set::exist(const Index &index) const { return _nodes.find(index) != _nodes.end(); }

void Set::iterate(const std::function<void(const Index &, const Node &)> &fn) const
{
  for (auto it = _nodes.begin(); it != _nodes.end(); ++it)
  {
    fn(it->first, *it->second);
  }
}

void Set::iterate(const std::function<void(const Index &, Node &)> &fn)
{
  for (auto it = _nodes.begin(); it != _nodes.end(); ++it)
  {
    fn(it->first, *it->second);
  }
}

} // namespace operation
} // namespace graph
} // namespace neurun
