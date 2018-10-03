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

#ifndef __NEURUN_GRAPH_OPERAND_SET_H__
#define __NEURUN_GRAPH_OPERAND_SET_H__

#include <memory>
#include <unordered_map>

#include "Object.h"
#include "Index.h"

namespace neurun
{
namespace graph
{
namespace operand
{

class Set
{
public:
  Set() : _index_count(0) {}

public:
  Index append(const Shape &, const TypeInfo &);

public:
  const Object &at(const Index &) const;
  Object &at(const Index &);
  bool exist(const Index &) const;
  void iterate(const std::function<void(const Index &, const Object &)> &fn) const;
  void iterate(const std::function<void(const Index &, Object &)> &fn);

private:
  const Index generateIndex();

private:
  std::unordered_map<Index, std::unique_ptr<Object>> _objects;
  uint32_t _index_count;
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_SET_H__
