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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_OPTIMIZED_OPS_H_
#define ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_OPTIMIZED_OPS_H_

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen matrix expression. The same explanation as for VectorMap
// above also applies here.
template <typename Scalar>
using MatrixMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar* data,
                                                const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar* data,
                                               const Dims<N>& dims) {
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++) {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename T>
inline void ExtractPatchIntoBufferColumn(
    const Dims<4>& input_dims, int w, int h, int b, int kheight, int kwidth,
    int stride_width, int stride_height, int pad_width, int pad_height,
    int in_width, int in_height, int in_depth, int single_buffer_length,
    int buffer_id, const T* in_data, T* conv_buffer_data, uint8 byte_zero) {
  gemmlowp::ScopedProfilingLabel label("ExtractPatchIntoBufferColumn");
  // This chunk of code reshapes all the inputs corresponding to
  // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
  const int kwidth_times_indepth = kwidth * in_depth;
  const int inwidth_times_indepth = in_width * in_depth;
  const int ih_ungated_start = h * stride_height - pad_height;
  const int ih_ungated_end = (ih_ungated_start + kheight);
  const int ih_end = std::min(ih_ungated_end, in_height);
  const int iw_ungated_start = w * stride_width - pad_width;
  const int iw_ungated_end = (iw_ungated_start + kwidth);
  const int iw_end = std::min(iw_ungated_end, in_width);
  // If the patch is off the edge of the input image, skip writing those rows
  // and columns from the patch into the output array.
  const int h_offset = std::max(0, -ih_ungated_start);
  const int w_offset = std::max(0, -iw_ungated_start);
  const int ih_start = std::max(0, ih_ungated_start);
  const int iw_start = std::max(0, iw_ungated_start);
  const int single_row_num =
      std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
  const int output_row_offset = (buffer_id * single_buffer_length);
  int out_offset =
      output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
  int in_offset = Offset(input_dims, 0, iw_start, ih_start, b);

  // Express all of the calculations as padding around the input patch.
  const int top_padding = h_offset;
  const int bottom_padding = (ih_ungated_end - ih_end);
  const int left_padding = w_offset;
  const int right_padding = (iw_ungated_end - iw_end);
  assert(single_row_num ==
         ((kwidth - (left_padding + right_padding)) * in_depth));

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0) {
    const int top_row_elements = (top_padding * kwidth * in_depth);
    memset(conv_buffer_data + output_row_offset, byte_zero,
           (top_row_elements * sizeof(T)));
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if ((left_padding == 0) && (right_padding == 0)) {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  } else {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      if (left_padding > 0) {
        const int left_start = (out_offset - (left_padding * in_depth));
        memset(conv_buffer_data + left_start, byte_zero,
               (left_padding * in_depth * sizeof(T)));
      }
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      if (right_padding > 0) {
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + right_start, byte_zero,
               (right_padding * in_depth * sizeof(T)));
      }
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0) {
    const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
    const int bottom_start =
        output_row_offset +
        ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
    memset(conv_buffer_data + bottom_start, byte_zero,
           (bottom_row_elements * sizeof(T)));
  }
}

#ifdef USE_NEON
template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float* bias_data,
                                      const Dims<4>& bias_dims,
                                      float* array_data,
                                      const Dims<4>& array_dims) {
  gemmlowp::ScopedProfilingLabel label("AddBiasAndEvalActivationFunction");
  const int bias_size = bias_dims.sizes[3] * bias_dims.strides[3];
  const int array_size = array_dims.sizes[3] * array_dims.strides[3];
  DCHECK_EQ((array_size % bias_size), 0);
  float* array_ptr = array_data;
  float* array_end_ptr = array_ptr + array_size;
  const auto zero = vdupq_n_f32(0);
  const auto six = vdupq_n_f32(6);
  const auto neg_one = vdupq_n_f32(-1);
  const auto one = vdupq_n_f32(1);
  for (; array_ptr != array_end_ptr; array_ptr += bias_size) {
    int i = 0;
    for (; i <= bias_size - 16; i += 16) {
      auto b0 = vld1q_f32(bias_data + i);
      auto b1 = vld1q_f32(bias_data + i + 4);
      auto b2 = vld1q_f32(bias_data + i + 8);
      auto b3 = vld1q_f32(bias_data + i + 12);
      auto a0 = vld1q_f32(array_ptr + i);
      auto a1 = vld1q_f32(array_ptr + i + 4);
      auto a2 = vld1q_f32(array_ptr + i + 8);
      auto a3 = vld1q_f32(array_ptr + i + 12);
      auto x0 = vaddq_f32(a0, b0);
      auto x1 = vaddq_f32(a1, b1);
      auto x2 = vaddq_f32(a2, b2);
      auto x3 = vaddq_f32(a3, b3);
      if (Ac == FusedActivationFunctionType::kRelu ||
          Ac == FusedActivationFunctionType::kRelu6) {
        x0 = vmaxq_f32(zero, x0);
        x1 = vmaxq_f32(zero, x1);
        x2 = vmaxq_f32(zero, x2);
        x3 = vmaxq_f32(zero, x3);
        if (Ac == FusedActivationFunctionType::kRelu6) {
          x0 = vminq_f32(six, x0);
          x1 = vminq_f32(six, x1);
          x2 = vminq_f32(six, x2);
          x3 = vminq_f32(six, x3);
        }
      } else if (Ac == FusedActivationFunctionType::kRelu1) {
        x0 = vmaxq_f32(neg_one, x0);
        x1 = vmaxq_f32(neg_one, x1);
        x2 = vmaxq_f32(neg_one, x2);
        x3 = vmaxq_f32(neg_one, x3);
        x0 = vminq_f32(one, x0);
        x1 = vminq_f32(one, x1);
        x2 = vminq_f32(one, x2);
        x3 = vminq_f32(one, x3);
      }
      vst1q_f32(array_ptr + i, x0);
      vst1q_f32(array_ptr + i + 4, x1);
      vst1q_f32(array_ptr + i + 8, x2);
      vst1q_f32(array_ptr + i + 12, x3);
    }
    for (; i <= bias_size - 4; i += 4) {
      auto b = vld1q_f32(bias_data + i);
      auto a = vld1q_f32(array_ptr + i);
      auto x = vaddq_f32(a, b);
      if (Ac == FusedActivationFunctionType::kRelu ||
          Ac == FusedActivationFunctionType::kRelu6) {
        x = vmaxq_f32(zero, x);
        if (Ac == FusedActivationFunctionType::kRelu6) {
          x = vminq_f32(six, x);
        }
      } else if (Ac == FusedActivationFunctionType::kRelu1) {
        x = vmaxq_f32(neg_one, x);
        x = vminq_f32(one, x);
      }
      vst1q_f32(array_ptr + i, x);
    }
    for (; i < bias_size; i++) {
      array_ptr[i] = ActivationFunction<Ac>(array_ptr[i] + bias_data[i]);
    }
  }
}
#else  // not NEON
template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float* bias_data,
                                      const Dims<4>& bias_dims,
                                      float* array_data,
                                      const Dims<4>& array_dims) {
  gemmlowp::ScopedProfilingLabel label("AddBiasAndEvalActivationFunction");
  const int bias_size = bias_dims.sizes[3] * bias_dims.strides[3];
  const int array_size = array_dims.sizes[3] * array_dims.strides[3];
  DCHECK_EQ((array_size % bias_size), 0);
  for (int array_offset = 0; array_offset < array_size;
       array_offset += bias_size) {
    for (int i = 0; i < bias_size; i++) {
      array_data[array_offset + i] =
          ActivationFunction<Ac>(array_data[array_offset + i] + bias_data[i]);
    }
  }
}
#endif

template <typename Lhs, typename Rhs, typename Result>
void Gemm(const Eigen::MatrixBase<Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs,
          Eigen::MatrixBase<Result>* result) {
  if (rhs.cols() == 1) {
    gemmlowp::ScopedProfilingLabel label("GEMV");
    result->col(0).noalias() = lhs * rhs.col(0);
  } else {
    gemmlowp::ScopedProfilingLabel label("GEMM");
    result->noalias() = lhs * rhs;
  }
}

template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride_width,
            int stride_height, int pad_width, int pad_height, int kheight,
            int kwidth, uint8 byte_zero, T* output_data,
            const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Im2col");
  DCHECK(IsPackedWithoutStrides(input_dims));
  DCHECK(IsPackedWithoutStrides(output_dims));
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        ExtractPatchIntoBufferColumn(
            input_dims, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            output_depth, buffer_id, input_data, output_data, byte_zero);
        ++buffer_id;
      }
    }
  }
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  (void)im2col_data;
  (void)im2col_dims;
  gemmlowp::ScopedProfilingLabel label("Conv");

  const float* gemm_input_data = nullptr;
  const Dims<4>* gemm_input_dims = nullptr;
  const int filter_width = ArraySize(filter_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_im2col) {
    DCHECK(im2col_data);
    Im2col(input_data, input_dims, stride_width, stride_height, pad_width,
           pad_height, filter_height, filter_width, 0, im2col_data,
           im2col_dims);
    gemm_input_data = im2col_data;
    gemm_input_dims = &im2col_dims;
  } else {
    DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_dims = &input_dims;
  }

  const auto im2col_matrix_map =
      MapAsMatrixWithFirstDimAsRows(gemm_input_data, *gemm_input_dims);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), im2col_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data,
                                       output_dims);
}

#endif  // ANDROID_ML_NN_COMMON_OPERATIONS_INTERNAL_OPTIMIZED_OPS_H_
