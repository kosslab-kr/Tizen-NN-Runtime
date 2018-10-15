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

#ifndef __NEURUN_GRAPH_OPERAND_SHAPE4D_CONVERT_H__
#define __NEURUN_GRAPH_OPERAND_SHAPE4D_CONVERT_H__

#include "LowerInfo.h"

namespace neurun
{
namespace graph
{
namespace operand
{

inline LowerInfo::Shape4D asShape4D(const Shape &shape)
{
  switch (shape.rank())
  {
    case 0u:
      return LowerInfo::Shape4D(1, 1, 1, 1);

    case 1u:
      return LowerInfo::Shape4D(1, 1, 1, shape.dim(0));

    case 2u:
      return LowerInfo::Shape4D(1, 1, shape.dim(1), shape.dim(0));

    case 3u:
      return LowerInfo::Shape4D(1, shape.dim(2), shape.dim(1), shape.dim(0));

    case 4u:
      return LowerInfo::Shape4D(shape.dim(3), shape.dim(2), shape.dim(1), shape.dim(0));

    default:
      throw "Unsupported rank > 4";
  }
}

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_SHAPE4D_CONVERT_H__
