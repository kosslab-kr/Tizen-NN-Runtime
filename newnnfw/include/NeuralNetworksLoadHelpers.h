/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __NEURAL_NETWORKS_LOAD_HELPER_H__
#define __NEURAL_NETWORKS_LOAD_HELPER_H__

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NNAPI_LOG(format, ...) printf(format "\n", __VA_ARGS__);
#define LOAD_FUNCTION(name)                                                    \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadNNAPIFunction(#name));
#define EXECUTE_FUNCTION(...)                                                  \
  if (fn != nullptr) {                                                         \
    fn(__VA_ARGS__);                                                           \
  }
#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

inline void* loadNNAPILibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
  return handle;
}

inline void* getNNAPILibraryHandle() {
  static void* handle = loadNNAPILibrary("libneuralnetworks.so");
  return handle;
}

inline void* loadNNAPIFunction(const char* name) {
  void* fn = nullptr;
  if (getNNAPILibraryHandle() != nullptr) {
    fn = dlsym(getNNAPILibraryHandle(), name);
  }
  if (fn == nullptr)
  {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
    abort();
  }
  else
  {
#ifdef _GNU_SOURCE
    Dl_info info;
    dladdr(fn, &info);
    NNAPI_LOG("nnapi function '%s' is loaded from '%s' ", name, info.dli_fname);
#endif // _GNU_SOURCE
  }
  return fn;
}

inline bool NNAPIExists() {
  static bool nnapi_is_available = getNNAPILibraryHandle();
  return nnapi_is_available;
}

#endif // __NEURAL_NETWORKS_LOAD_HELPER_H__
