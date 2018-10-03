/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __INTERNAL_ARM_COMPUTE_H__
#define __INTERNAL_ARM_COMPUTE_H__

#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/Tensor.h>

namespace internal
{
namespace arm_compute
{
namespace operand
{

class Object
{
public:
  Object() = default;

public:
  Object(const std::shared_ptr<::arm_compute::ITensor> &tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

public:
  ::arm_compute::ITensor *ptr(void) const { return _tensor.get(); }

private:
  std::shared_ptr<::arm_compute::ITensor> _tensor;

public:
  void access(const std::function<void(::arm_compute::ITensor &tensor)> &fn) const;
};

} // namespace operand
} // namepsace arm_compute
} // namespace internal

#include "internal/Model.h"

#include <map>

namespace internal
{
namespace arm_compute
{
namespace operand
{

class Context
{
public:
  Context &set(const ::internal::tflite::operand::Index &ind,
               const std::shared_ptr<::arm_compute::ITensor> &tensor);

public:
  bool exist(const ::internal::tflite::operand::Index &ind) const
  {
    return _objects.find(ind.asInt()) != _objects.end();
  }

public:
  const Object &at(const ::internal::tflite::operand::Index &ind) const
  {
    return _objects.at(ind.asInt());
  }

  Object &at(const ::internal::tflite::operand::Index &ind) { return _objects.at(ind.asInt()); }

private:
  std::map<int, Object> _objects;
};

} // namespace operand
} // namepsace arm_compute
} // namespace internal

#include <arm_compute/runtime/IFunction.h>

namespace internal
{
namespace arm_compute
{
namespace op
{

class Step
{
public:
  Step(std::unique_ptr<::arm_compute::IFunction> &&func) : _func{std::move(func)}
  {
    // DO NOTHING
  }

public:
  void run(void) const { _func->run(); }

public:
  const std::string &name(void) const { return _name; }
  std::string &name(void) { return _name; }

private:
  std::string _name;
  std::unique_ptr<::arm_compute::IFunction> _func;
#ifdef TFLITE_PROFILING_ENABLED
public:
  int op_idx() const { return _op_idx; }
  int &op_idx() { return _op_idx; }
private:
  int _op_idx;
#endif
};

} // namespace op
} // namepsace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{
namespace op
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
  Step &at(uint32_t n) { return _functions.at(n); }
  const Step &at(uint32_t n) const { return _functions.at(n); }

private:
  // TODO Rename _functions as _steps
  std::vector<Step> _functions;
};

} // namespace op
} // namepsace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{

class Plan
{
public:
  Plan(const std::shared_ptr<const ::internal::tflite::Model> &model) : _model(model)
  {
    // DO NOTHING
  }

public:
  const ::internal::tflite::Model &model(void) const { return *_model; }

public:
  operand::Context &operands(void) { return _operands; }
  const operand::Context &operands(void) const { return _operands; }

public:
  op::Sequence &operations(void) { return _ops; }
  const op::Sequence &operations(void) const { return _ops; }

private:
  std::shared_ptr<const ::internal::tflite::Model> _model;
  operand::Context _operands;
  op::Sequence _ops;
};

} // namepsace arm_compute
} // namespace internal

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{

// check if this runtime runs on GPU or NEON
bool isGpuMode();

#define CAST_CL(tensor) static_cast<::arm_compute::CLTensor *>(tensor)
#define CAST_NE(tensor) static_cast<::arm_compute::Tensor *>(tensor)

} // namepsace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_H__
