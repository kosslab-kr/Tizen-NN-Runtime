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

#ifndef __NEURUN_CODEGEN_OPERAND_CONTEXT_H__
#define __NEURUN_CODEGEN_OPERAND_CONTEXT_H__

#include "backend/IObject.h"
#include "graph/operand/Index.h"

#include <map>

namespace neurun
{
namespace codegen
{
namespace operand
{

class Context
{
public:
  Context &set(const graph::operand::Index &ind,
               const std::shared_ptr<backend::operand::IObject> &object);

public:
  bool exist(const ::neurun::graph::operand::Index &ind) const
  {
    return _objects.find(ind.asInt()) != _objects.end();
  }

public:
  const std::vector<std::shared_ptr<backend::operand::IObject>> &
  at(const graph::operand::Index &ind) const
  {
    return _objects.at(ind.asInt());
  }

  std::vector<std::shared_ptr<backend::operand::IObject>> &at(const graph::operand::Index &ind)
  {
    return _objects.at(ind.asInt());
  }

private:
  std::map<int, std::vector<std::shared_ptr<backend::operand::IObject>>> _objects;
};

} // namespace operand
} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_OPERAND_CONTEXT_H__
