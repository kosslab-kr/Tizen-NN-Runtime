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

#ifndef __NEURUN_CODEGEN_OPERATION_SEQUENCE_H__
#define __NEURUN_CODEGEN_OPERATION_SEQUENCE_H__

#include <stdint.h>
#include <arm_compute/runtime/IFunction.h>
#include <memory>
#include <vector>

namespace neurun
{
namespace codegen
{
namespace operation
{

class Sequence
{
public:
  uint32_t size(void) const { return _functions.size(); }

public:
  Sequence &append(std::unique_ptr<::arm_compute::IFunction> &&func)
  {
    _functions.emplace_back(std::move(func));
    return (*this);
  }

public:
  ::arm_compute::IFunction &at(uint32_t n) const { return *(_functions.at(n)); }

private:
  std::vector<std::unique_ptr<::arm_compute::IFunction>> _functions;
};

} // namespace operation
} // namespace codegen
} // namespace neurun

#endif // __NEURUN_CODEGEN_OPERATION_SEQUENCE_H__
