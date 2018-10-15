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

#include "Node.h"

#include "LowerInfo.h"

namespace neurun
{
namespace graph
{
namespace operation
{

Node::Node() = default;

Node::~Node() = default;

void Node::lower_info(std::unique_ptr<LowerInfo> &&lower_info)
{
  _lower_info = std::move(lower_info);
}

const LowerInfo *Node::lower_info() const { return _lower_info.get(); }

} // namespace operation
} // namespace graph
} // namespace neurun
