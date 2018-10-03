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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_TYPES_H_
#define ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_TYPES_H_

enum class OperandType : int32_t {
    FLOAT32 = 0,
    INT32 = 1,
    UINT32 = 2,
    TENSOR_FLOAT32 = 3,
    TENSOR_INT32 = 4,
    TENSOR_QUANT8_ASYMM = 5,
    OEM = 10000,
    TENSOR_OEM_BYTE = 10001,
};

#include "compatibility.h"

enum class FusedActivationFunctionType { kNone, kRelu6, kRelu1, kRelu };

template <int N>
struct Dims {
  int sizes[N];
  int strides[N];
};

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

inline uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx) {
    if (dimensionIdx >= shape.dimensions.size()) {
        // TODO, log the error
        return 0;
    }
    return shape.dimensions[dimensionIdx];
}

inline Dims<4> convertShapeToDims(const Shape& shape) {
  Dims<4> dims;
  for (int i=0; i<4; i++) {
    dims.sizes[i] = 1;
  }

  if (shape.dimensions.size() == 1) {
    dims.sizes[0] = (int)getSizeOfDimension(shape, 0);
  } else {
    for (int i=0; i<4; i++) {
      int src = (int)shape.dimensions.size()-i-1;
      if (src >= 0) {
        dims.sizes[i] = (int)getSizeOfDimension(shape, src);
      }
    }
  }

  dims.strides[0] = 1;
  for (int i = 1; i<4; i++) {
    dims.strides[i] = dims.strides[i-1] * dims.sizes[i-1];
  }
  return dims;
}

inline int Offset(const Dims<4>& dims, int i0, int i1, int i2, int i3) {
  DCHECK(i0 >= 0 && i0 < dims.sizes[0]);
  DCHECK(i1 >= 0 && i1 < dims.sizes[1]);
  DCHECK(i2 >= 0 && i2 < dims.sizes[2]);
  DCHECK(i3 >= 0 && i3 < dims.sizes[3]);
  return i0 * dims.strides[0] + i1 * dims.strides[1] + i2 * dims.strides[2] +
         i3 * dims.strides[3];
}

// Get array size, DCHECKing that the dim index is in range.
template <int N>
int ArraySize(const Dims<N>& array, int index) {
  DCHECK(index >= 0 && index < N);
  return array.sizes[index];
}

// Get common array size, DCHECKing that they all agree.
template <typename ArrayType1, typename ArrayType2>
int MatchingArraySize(const ArrayType1& array1, int index1,
                      const ArrayType2& array2, int index2) {
  DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return ArraySize(array1, index1);
}

template <typename ArrayType1, typename ArrayType2, typename... Args>
int MatchingArraySize(const ArrayType1& array1, int index1,
                      const ArrayType2& array2, int index2, Args... args) {
  DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return MatchingArraySize(array1, index1, args...);
}

inline int RequiredBufferSizeForDims(const Dims<4>& dims) {
  int max_offset = 0;
  for (int i = 0; i < 4; i++) {
    max_offset += (dims.sizes[i] - 1) * dims.strides[i];
  }
  return max_offset + 1;
}

template <int N>
bool IsPackedWithoutStrides(const Dims<N>& dims) {
  int expected_stride = 1;
  for (int d = 0; d < N; d++) {
    if (dims.strides[d] != expected_stride) return false;
    expected_stride *= dims.sizes[d];
  }
  return true;
}

#endif  // ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_TYPES_H_
