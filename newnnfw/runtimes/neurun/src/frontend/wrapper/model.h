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

#ifndef __MODEL_H__
#define __MODEL_H__

#include <NeuralNetworks.h>

#include "graph/Graph.h"

struct ANeuralNetworksModel
{
public:
  ANeuralNetworksModel();

public:
  neurun::graph::Graph &deref(void) { return *_model; }
  ResultCode finish();
  bool isFinished() { return !_model->isBuildingPhase(); }

public:
  void release(std::shared_ptr<neurun::graph::Graph> &model) { model = _model; }

private:
  std::shared_ptr<neurun::graph::Graph> _model;
};

#endif // __MODEL_H__
