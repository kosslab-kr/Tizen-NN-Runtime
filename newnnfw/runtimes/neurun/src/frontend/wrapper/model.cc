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

#include "model.h"

#include "graph/Graph.h"

//
// ANeuralNetworksModel
//
ANeuralNetworksModel::ANeuralNetworksModel() : _model{new neurun::graph::Graph}
{
  // DO NOTHING
}

ResultCode ANeuralNetworksModel::finish()
{
  // This function must only be called once for a given model
  if (isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  _model->finishBuilding();

  return ANEURALNETWORKS_NO_ERROR;
}
