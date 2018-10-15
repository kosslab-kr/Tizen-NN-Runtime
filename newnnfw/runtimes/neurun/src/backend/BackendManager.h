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

#ifndef __NEURUN_BACKEND_BACKEND_MANAGER_H__
#define __NEURUN_BACKEND_BACKEND_MANAGER_H__

#include <memory>
#include <map>

#include "graph/operand/Set.h"

namespace neurun
{
namespace backend
{

struct IBackendConfig;
struct IInitializerGenerator;
struct IStageGenerator;
struct ITensorBuilder;

class Backend
{
public:
  Backend(const std::shared_ptr<neurun::backend::IBackendConfig> &backend_config,
          const std::shared_ptr<neurun::backend::IInitializerGenerator> &initializer_gen,
          const std::shared_ptr<neurun::backend::IStageGenerator> &stage_gen);

  Backend(void) : _config(nullptr), _initializer_gen(nullptr), _stage_gen(nullptr)
  {
    // DO NOTHING
  }

public:
  const std::shared_ptr<neurun::backend::IBackendConfig> config() const;
  const std::shared_ptr<neurun::backend::IInitializerGenerator> initializer_gen() const;
  const std::shared_ptr<neurun::backend::IStageGenerator> stage_gen() const;
  const std::shared_ptr<neurun::backend::ITensorBuilder> tensor_builder() const;

private:
  std::shared_ptr<neurun::backend::IBackendConfig> _config;
  std::shared_ptr<neurun::backend::IInitializerGenerator> _initializer_gen;
  std::shared_ptr<neurun::backend::IStageGenerator> _stage_gen;
};

class BackendManager
{
public:
  BackendManager(const neurun::graph::operand::Set &operands);

  Backend get(const std::string &key);

private:
  std::map<std::string, Backend> _gen_map;
};

} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_BACKEND_MANAGER_H__
