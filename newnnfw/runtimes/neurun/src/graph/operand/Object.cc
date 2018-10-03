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

#include "Object.h"

namespace neurun
{
namespace graph
{
namespace operand
{

size_t Object::operandSize(void) const
{
  const uint32_t ranks = _shape.rank();
  int32_t elements = 1;

  for (uint32_t rank = 0; rank < ranks; rank++)
  {
    elements *= _shape.dim(rank);
  }

  DataType type = _type.type();
  size_t element_size = 0;

  // Value of type is matched with OperandCode enum in NeuralNetworks.h
  switch (type)
  {
    case DataType::SCALAR_FLOAT32:
    case DataType::TENSOR_FLOAT32:
      element_size = sizeof(float);
      break;
    case DataType::SCALAR_INT32:
    case DataType::TENSOR_INT32:
      element_size = sizeof(int32_t);
      break;
    case DataType::SCALAR_UINT32:
      element_size = sizeof(uint32_t);
      break;
    case DataType::TENSOR_QUANT8_ASYMM:
      element_size = sizeof(uint8_t);
      break;
    default:
      throw std::runtime_error{"Unsuppported type size"};
  }

  return element_size * elements;
}

bool Object::setUsage(const OperandUsage usage)
{
  if (usageIsDefined() && (_usage != usage))
  {
    // Already set as different type
    return false;
  }

  _usage = usage;

  return true;
}

void Object::appendUse(const ::neurun::graph::operation::Index &idx)
{
  assert(_usage != OperandUsage::NOT_DEFINED);
  assert(!_uses.contains(idx));

  _uses.append(idx);
}

void Object::removeUse(const ::neurun::graph::operation::Index &idx)
{
  assert(_usage != OperandUsage::NOT_DEFINED);
  assert(_uses.contains(idx));

  _uses.remove(idx);
}

void Object::appendDef(const ::neurun::graph::operation::Index &idx)
{
  assert(_usage != OperandUsage::NOT_DEFINED && _usage != OperandUsage::CONSTANT);
  assert(_def.size() == 0);

  _def.append(idx);
}

void Object::removeDef(const ::neurun::graph::operation::Index &idx)
{
  assert(_usage != OperandUsage::NOT_DEFINED);
  assert(_def.contains(idx));

  _def.remove(idx);
}

void Object::lower_info(std::unique_ptr<LowerInfo> &&lower_info)
{
  _lower_info = std::move(lower_info);
}

const LowerInfo *Object::lower_info() const { return _lower_info.get(); }

} // namespace operand
} // namespace graph
} // namespace neurun
