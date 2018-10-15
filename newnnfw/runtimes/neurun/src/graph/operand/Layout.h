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

#ifndef __NEURUN_GRAPH_OPERAND_LAYOUT_H__
#define __NEURUN_GRAPH_OPERAND_LAYOUT_H__

#include <functional>

namespace neurun
{
namespace graph
{
namespace operand
{

enum class Layout
{
  UNKNOWN = 0,
  NHWC,
  NCHW
};

} // namespace operand
} // namespace graph
} // namespace neurun

namespace std
{

template <> struct hash<::neurun::graph::operand::Layout>
{
  size_t operator()(const ::neurun::graph::operand::Layout &value) const noexcept
  {
    using type = typename std::underlying_type<::neurun::graph::operand::Layout>::type;
    return hash<type>()(static_cast<type>(value));
  }
};

} // namespace std

#endif // __NEURUN_GRAPH_OPERAND_LAYOUT_H__
