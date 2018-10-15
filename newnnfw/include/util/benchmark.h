/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_UTIL_BENCHMARK_H__
#define __NNFW_UTIL_BENCHMARK_H__

#include <chrono>

namespace nnfw
{
namespace util
{
// Benckmark support
namespace benchmark
{

template <typename T> class Accumulator
{
public:
  Accumulator(T &ref) : _ref(ref)
  {
    // DO NOTHING
  }

public:
  T &operator()(void) { return _ref; }

private:
  T &_ref;
};

template <typename T, typename Callable>
Accumulator<T> &operator<<(Accumulator<T> &&acc, Callable cb)
{
  auto begin = std::chrono::steady_clock::now();
  cb();
  auto end = std::chrono::steady_clock::now();

  acc() += std::chrono::duration_cast<T>(end - begin);

  return acc;
}

template <typename T> Accumulator<T> measure(T &out) { return Accumulator<T>(out); }

} // namespace benchmark
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_BENCHMARK_H__
