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

#ifndef __COMPILATION_H__
#define __COMPILATION_H__

#include "internal/Model.h"
#include "internal/arm_compute.h"

struct ANeuralNetworksCompilation
{
public:
  ANeuralNetworksCompilation(const std::shared_ptr<const internal::tflite::Model> &model)
      : _plan{new internal::arm_compute::Plan{model}}
  {
    // DO NOTHING
  }

public:
  internal::arm_compute::Plan &plan(void) { return *_plan; }

public:
  void publish(std::shared_ptr<const internal::arm_compute::Plan> &plan) { plan = _plan; }
  bool isFinished(void) { return _isFinished; }
  void markAsFinished() { _isFinished = true; }

private:
  std::shared_ptr<internal::arm_compute::Plan> _plan;
  bool _isFinished{false};
};

#endif
