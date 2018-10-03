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

#ifndef __EXECUTION_H__
#define __EXECUTION_H__

#include "internal/arm_compute.h"
#include "internal/Sink.h"
#include "internal/Source.h"

struct ANeuralNetworksExecution
{
public:
  ANeuralNetworksExecution(const std::shared_ptr<const internal::arm_compute::Plan> &plan)
      : _plan{plan}
  {
    _sources.resize(_plan->model().inputs.size());
    _sinks.resize(_plan->model().outputs.size());
  }

public:
  const internal::arm_compute::Plan &plan(void) const { return *_plan; }

private:
  std::shared_ptr<const internal::arm_compute::Plan> _plan;

public:
  // TODO Use InputIndex instead of int
  void source(int n, std::unique_ptr<Source> &&source) { _sources.at(n) = std::move(source); }
  template <typename T, typename... Args> void source(int n, Args &&... args)
  {
    source(n, std::unique_ptr<T>{new T{std::forward<Args>(args)...}});
  }

public:
  const Source &source(int n) const { return *(_sources.at(n)); }

public:
  // TODO Use OutputIndex instead of int
  void sink(int n, std::unique_ptr<Sink> &&sink) { _sinks.at(n) = std::move(sink); }
  template <typename T, typename... Args> void sink(int n, Args &&... args)
  {
    sink(n, std::unique_ptr<T>{new T{std::forward<Args>(args)...}});
  }

public:
  const Sink &sink(int n) const { return *(_sinks.at(n)); }

private:
  std::vector<std::unique_ptr<Source>> _sources;
  std::vector<std::unique_ptr<Sink>> _sinks;
};

#endif
