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

#ifndef __NEURUN_BACKEND_CPU_BACKEND_CONFIG_H__
#define __NEURUN_BACKEND_CPU_BACKEND_CONFIG_H__

#include "backend/IBackendConfig.h"

namespace neurun
{
namespace backend
{
namespace cpu
{

class BackendConfig : public IBackendConfig
{
public:
  BackendConfig()
  {
    // DO NOTHING
  }

  virtual void initialize() override;
  virtual graph::operand::Layout getOperandLayout() { return graph::operand::Layout::NHWC; }
};

} // namespace cpu
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_CPU_BACKEND_CONFIG_H__
