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

#include <NeuralNetworks.h>

#include <new>

#include "frontend/wrapper/model.h"
#include "frontend/wrapper/compilation.h"

//
// NNAPI Implementation
//
int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_create start /n");
  if ((model == nullptr) || (compilation == nullptr))
  {
    printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_create return UNEXPECTED_NULL/n");
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  std::shared_ptr<neurun::graph::Graph> internal;

  model->release(internal);

  *compilation = new (std::nothrow) ANeuralNetworksCompilation(internal);
  if (*compilation == nullptr)
  {
    printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_create return OUT_OF_MEMORY/n");
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_create return/n");
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_finish start /n");
  if (compilation == nullptr)
  {
    printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_finish return UNEXPECTED_NULL/n");
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_finish return/n");
  return compilation->finish();
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_free start /n");
  delete compilation;
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_free return/n");
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                             int32_t /* preference */)
{
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_setPreference start /n");
  if (compilation == nullptr)
  {
    printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_setPreference return UNEXPECTED_NULL /n");
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // NYi
  printf("nnfw/runtimes/neurun/src/frontend/compilation.cc -----> ANeuralNetworksCompilation_setPreference return /n");
  return ANEURALNETWORKS_NO_ERROR;
}
