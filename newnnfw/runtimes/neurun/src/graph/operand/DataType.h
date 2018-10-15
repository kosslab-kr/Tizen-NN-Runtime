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

#ifndef __NEURUN_GRAPH_OPERAND_DATATYPE_H__
#define __NEURUN_GRAPH_OPERAND_DATATYPE_H__

namespace neurun
{
namespace graph
{
namespace operand
{

enum class DataType
{
  SCALAR_FLOAT32 = 0,
  SCALAR_INT32 = 1,
  SCALAR_UINT32 = 2,

  TENSOR_FLOAT32 = 3,
  TENSOR_INT32 = 4,

  TENSOR_QUANT8_ASYMM = 5,
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_DATATYPE_H__
