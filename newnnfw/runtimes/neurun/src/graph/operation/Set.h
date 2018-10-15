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

#ifndef __NEURUN_GRAPH_OPERATION_SET_H__
#define __NEURUN_GRAPH_OPERATION_SET_H__

#include <memory>

#include "graph/operation/Index.h"
#include "Node.h"

#include <unordered_map>

namespace neurun
{
namespace graph
{
namespace operation
{

class Set
{
public:
  Set() : _index_count(0) {}

public:
  Index append(std::unique_ptr<Node> &&node);

public:
  const Node &at(const Index &) const;
  Node &at(const Index &);
  bool exist(const Index &) const;
  uint32_t size() const { return _nodes.size(); }
  void iterate(const std::function<void(const Index &, const Node &)> &fn) const;
  void iterate(const std::function<void(const Index &, Node &)> &fn);

private:
  const Index generateIndex();

private:
  std::unordered_map<Index, std::unique_ptr<Node>> _nodes;
  uint32_t _index_count;
};

} // namespace operation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_SET_H__
