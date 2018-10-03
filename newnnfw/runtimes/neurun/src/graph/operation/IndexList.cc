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

#include "IndexList.h"

#include <algorithm>

namespace neurun
{
namespace graph
{
namespace operation
{

IndexList::IndexList(std::initializer_list<Index> list) : _list(list)
{
  // DO NOTHING
}

bool IndexList::contains(const ::neurun::graph::operation::Index &index) const
{
  return std::find(_list.begin(), _list.end(), index) != _list.end();
}

} // namespace operation
} // namespace graph
} // namespace neurun
