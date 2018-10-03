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

#ifndef __NEURUN_GRAPH_OPERAND_INDEX_H__
#define __NEURUN_GRAPH_OPERAND_INDEX_H__

#include "graph/Index.h"

namespace neurun
{
namespace graph
{
namespace operand
{

using Index = ::neurun::graph::Index<uint32_t, struct IndexTag>;

} // namespace operand
} // namespace graph
} // namespace neurun

namespace neurun
{
namespace graph
{
namespace operand
{
namespace IO
{

using Index = ::neurun::graph::Index<uint32_t, struct IndexTag>;

} // namespace IO
} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_INDEX_H__
