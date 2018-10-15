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

#ifndef __NEURUN_GRAPH_INDEX_H__
#define __NEURUN_GRAPH_INDEX_H__

#include <functional>
#include <stdint.h>

namespace neurun
{
namespace graph
{

template <typename T, typename DummyTag> class Index
{
public:
  explicit Index(T o) : _index{o} {}
  explicit Index(int32_t o) : _index{static_cast<T>(o)} {} // For legacy code compatibility
  Index(const Index &o) : _index{o._index} {}

  Index &operator=(T o)
  {
    _index = o;
    return *this;
  }

  Index &operator=(const T &o)
  {
    _index = o._index;
    return *this;
  }

  bool operator==(T o) const { return _index == o; }
  bool operator==(const Index &o) const { return _index == o._index; }
  bool operator!=(T o) const { return !(*this == o); }
  bool operator!=(const Index &o) const { return !(*this == o); }

  T value() const { return _index; }
  int32_t asInt() const { return static_cast<int32_t>(_index); } // For legacy code compatibility

private:
  T _index;
};

} // namespace graph
} // namespace neurun

namespace std
{

template <typename T, typename Tag> struct hash<::neurun::graph::Index<T, Tag>>
{
  size_t operator()(const ::neurun::graph::Index<T, Tag> &index) const noexcept
  {
    return hash<T>()(index.value());
  }
};

} // namespace std

#endif // __NEURUN_GRAPH_INDEX_H__
