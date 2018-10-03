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

/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
#define ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H

#include <cstdint>
#include <vector>
#include <ostream>

#define LOG(ERROR) std::cerr

// Macro to check if the input parameters for operation are valid or not.
#define NN_CHECK(v)                                                     \
  do {                                                                  \
    if (!(v)) {                                                         \
      LOG(ERROR) << "NN_CHECK failed: "  << #v << "'\n";                \
      return false;                                                     \
    }                                                                   \
  } while(0);

#define NN_CHECK_EQ(actual, expected)           \
  NN_CHECK((actual) == (expected))

#define NN_OPS_CHECK NN_CHECK

enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};

enum class FusedActivationFunc : int32_t {
    NONE = 0,
    RELU = 1,
    RELU1 = 2,
    RELU6 = 3,
};


#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            LOG(ERROR) << "Unsupported fused activation function type";     \
            return false;                                                   \
    }


#endif // ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
