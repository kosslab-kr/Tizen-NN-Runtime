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

#include "BackendManager.h"

#include "backend/acl_cl/BackendConfig.h"
#include "backend/acl_cl/TensorBuilder.h"
#include "backend/acl_cl/InitializerGenerator.h"
#include "backend/acl_cl/StageGenerator.h"
#include "backend/cpu/BackendConfig.h"
#include "backend/cpu/TensorBuilder.h"
#include "backend/cpu/InitializerGenerator.h"
#include "backend/cpu/StageGenerator.h"

namespace neurun
{
namespace backend
{

Backend::Backend(const std::shared_ptr<neurun::backend::IBackendConfig> &backend_config,
                 const std::shared_ptr<neurun::backend::IInitializerGenerator> &initializer_gen,
                 const std::shared_ptr<neurun::backend::IStageGenerator> &stage_gen)
    : _config(backend_config), _initializer_gen(initializer_gen), _stage_gen(stage_gen)
{
  backend_config->initialize();
}

const std::shared_ptr<neurun::backend::IBackendConfig> Backend::config() const { return _config; }

const std::shared_ptr<neurun::backend::IInitializerGenerator> Backend::initializer_gen() const
{
  return _initializer_gen;
}

const std::shared_ptr<neurun::backend::IStageGenerator> Backend::stage_gen() const
{
  return _stage_gen;
}

const std::shared_ptr<neurun::backend::ITensorBuilder> Backend::tensor_builder() const
{
  return _stage_gen->tensor_builder();
}

BackendManager::BackendManager(const neurun::graph::operand::Set &operands)
{
  // Add arm_compute backend
  {
    using namespace ::neurun::backend::acl_cl;
    auto acl_backend_initializer = std::make_shared<BackendConfig>();
    auto acl_tensor_builder = std::make_shared<TensorBuilder>();
    auto acl_initializer_gen = std::make_shared<InitializerGenerator>(operands);
    auto acl_stage_gen = std::make_shared<StageGenerator>(operands, acl_tensor_builder);

    // TODO Do not use magic string for backend id
    _gen_map["acl_cl"] = {acl_backend_initializer, acl_initializer_gen, acl_stage_gen};
  }

  // Add CPU backend
  {
    using namespace ::neurun::backend::cpu;
    auto cpu_backend_initializer = std::make_shared<BackendConfig>();
    auto cpu_tensor_builder = std::make_shared<TensorBuilder>();
    auto cpu_initializer_gen = std::make_shared<InitializerGenerator>(operands);
    auto cpu_stage_gen = std::make_shared<StageGenerator>(operands, cpu_tensor_builder);

    // TODO Do not use magic string for backend id
    _gen_map["cpu"] = {cpu_backend_initializer, cpu_initializer_gen, cpu_stage_gen};
  }
}

Backend BackendManager::get(const std::string &key) { return _gen_map.at(key); }

} // namespace backend
} // namespace neurun
