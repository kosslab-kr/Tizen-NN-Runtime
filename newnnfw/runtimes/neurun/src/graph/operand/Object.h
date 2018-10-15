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

#ifndef __NEURUN_GRAPH_OPERAND_OBJECT_H__
#define __NEURUN_GRAPH_OPERAND_OBJECT_H__

#include <cassert>
#include <cstdint>
#include <memory>
#include <algorithm>

#include "Shape.h"
#include "Data.h"
#include "TypeInfo.h"
#include "LowerInfo.h"
#include "graph/operation/IndexList.h"

namespace neurun
{
namespace graph
{
namespace operand
{

// Operand usage should be exact one of these
enum class OperandUsage
{
  NOT_DEFINED,
  MODEL_INPUT,
  CONSTANT,
  OPERATION_OUTPUT,
};

class Object
{
public:
  explicit Object(const Shape &shape, const TypeInfo &type)
      : _shape{shape}, _type{type}, _usage{OperandUsage::NOT_DEFINED}
  {
    // DO NOTHING
  }

public:
  const Shape &shape(void) const { return _shape; }
  const TypeInfo &typeInfo(void) const { return _type; }
  size_t operandSize(void) const;
  bool setAsConstant() { return setUsage(OperandUsage::CONSTANT); }
  bool setAsModelInput() { return setUsage(OperandUsage::MODEL_INPUT); }
  bool setAsOperationOutput() { return setUsage(OperandUsage::OPERATION_OUTPUT); }
  bool usageIsDefined(void) const { return _usage != OperandUsage::NOT_DEFINED; }
  bool isModelInput(void) const { return _usage == OperandUsage::MODEL_INPUT; }

  const operation::IndexList &getUses() const { return _uses; }
  const operation::IndexList &getDef() const { return _def; }
  void appendUse(const operation::Index &idx);
  void removeUse(const operation::Index &idx);
  void appendDef(const operation::Index &idx);
  void removeDef(const operation::Index &idx);

private:
  bool setUsage(OperandUsage usage);

public:
  void data(std::unique_ptr<Data> &&data) { _data = std::move(data); }
  const Data &data(void) const { return *_data; }

public:
  template <typename T, typename... Args> void data(Args &&... args)
  {
    data(std::unique_ptr<T>(new T{std::forward<Args>(args)...}));
  }

public:
  template <typename T> T asScalar(void) const
  {
    assert((_shape.rank() == 0) || ((_shape.rank() == 1) && (_shape.dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return *(reinterpret_cast<const T *>(_data->base()));
  }

public:
  void lower_info(std::unique_ptr<LowerInfo> &&lower_info);
  const LowerInfo *lower_info() const;

private:
  const Shape _shape;
  const TypeInfo _type;
  std::unique_ptr<Data> _data;
  OperandUsage _usage;

  operation::IndexList _uses;
  operation::IndexList _def; // size is 0 (constant) or 1 (from def operation)

  std::unique_ptr<LowerInfo> _lower_info;
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_OBJECT_H__
