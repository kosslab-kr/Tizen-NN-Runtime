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

#ifndef __NEURUN_CODEGEN_BACKEND_RESOLVER_H__
#define __NEURUN_CODEGEN_BACKEND_RESOLVER_H__

#include <set>
#include <unordered_map>
#include <typeindex>

#include "logging.h"
#include "util/EnvVar.h"
#include "backend/BackendManager.h"
#include "backend/IInitializerGenerator.h"
#include "backend/IStageGenerator.h"

namespace neurun
{
namespace codegen
{

class BackendResolver
{
public:
  BackendResolver(const neurun::graph::operand::Set &operands)
  {
    _backend_manager = std::make_shared<backend::BackendManager>(operands);

    const auto &backend_all_str =
        ::nnfw::util::EnvVar{std::string("OP_BACKEND_ALLOPS")}.asString("none");
    if (backend_all_str.compare("none") != 0)
    {
      VERBOSE(BackendResolver) << "Use backend for all ops: " << backend_all_str << std::endl;
#define OP(InternalName, NnApiName)                                   \
  {                                                                   \
    auto backend = _backend_manager->get(backend_all_str);            \
    _gen_map[typeid(graph::operation::InternalName::Node)] = backend; \
  }
#include "graph/operation/Op.lst"
#undef OP
    }
    else
    {
#define OP(InternalName, NnApiName)                                                               \
  {                                                                                               \
    const auto &backend_str =                                                                     \
        ::nnfw::util::EnvVar{std::string("OP_BACKEND_") + #NnApiName}.asString("acl_cl");         \
    auto backend = _backend_manager->get(backend_str);                                            \
    VERBOSE(BackendResolver) << "backend for " << #NnApiName << ": " << backend_str << std::endl; \
    _gen_map[typeid(graph::operation::InternalName::Node)] = backend;                             \
  }

#include "graph/operation/Op.lst"
#undef OP
    }
  }

public:
  const backend::Backend &getBackend(const std::type_index &type) { return _gen_map[type]; }

private:
  std::unordered_map<std::type_index, backend::Backend> _gen_map;
  std::shared_ptr<backend::BackendManager> _backend_manager;
};

} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_BACKEND_RESOLVER_H__
