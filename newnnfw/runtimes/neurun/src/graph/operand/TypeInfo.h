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

#ifndef __NEURUN_GRAPH_OPERAND_TYPEINFO_H__
#define __NEURUN_GRAPH_OPERAND_TYPEINFO_H__

#include <cstdint>

#include <NeuralNetworks.h>

#include "DataType.h"

namespace neurun
{
namespace graph
{
namespace operand
{

class TypeInfo
{
public:
  TypeInfo(OperandCode type, float scale, int32_t offset)
      : _type(typeFromOperandCode(type)), _scale(scale), _offset(offset)
  {
    // DO NOTHING
  }

public:
  DataType type() const { return _type; }
  float scale() const { return _scale; }
  int32_t offset() const { return _offset; }

private:
  // Now neurun::graph::operand::DataType share same enum value with OperandCode
  // in NeuralNetworks.h.
  // If we don't share same value, we must fix this mapping function.
  DataType typeFromOperandCode(OperandCode type);

private:
  DataType _type;
  float _scale;
  int32_t _offset;
};
} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_TYPEINFO_H__
