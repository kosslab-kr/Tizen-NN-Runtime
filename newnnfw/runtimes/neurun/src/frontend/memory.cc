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
#include <sys/mman.h>
#include <new>
#include <memory>

#include "nnfw/std/memory.h"
#include "frontend/wrapper/memory.h"

int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                       ANeuralNetworksMemory **memory)
{
  printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_createFromFd start /n");
  if (memory == nullptr)
  {
    printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_createFromFd return UNEXPECTED_NULL /n");
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // Use unique pointer to avoid memory leak
  std::unique_ptr<ANeuralNetworksMemory> memory_ptr =
      nnfw::make_unique<ANeuralNetworksMemory>(size, protect, fd, offset);
  if (memory_ptr == nullptr)
  {
    printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_createFromFd return OUT_OF_MEMORY /n");
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *memory = memory_ptr.release();

  printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_createFromFd return /n");
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory *memory)
{
  printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_free start /n");
  delete memory;
  printf("nnfw/runtimes/neurun/src/frontend/memory.cc -----> ANeuralNetworksMemory_free return /n");
}
