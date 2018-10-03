/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLTOPK_V2_H__
#define __ARM_COMPUTE_CLTOPK_V2_H__

#include "arm_compute/core/CL/kernels/CLTopKV2Kernel.h"

#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute TopK operation. This function calls the following OpenCL kernels:
 *
 * -# @ref CLTopKV2Kernel
 */
class CLTopKV2 : public IFunction
{
public:
  /** Constructor */
  CLTopKV2();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2(const CLTopKV2 &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2 &operator=(const CLTopKV2 &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2(CLTopKV2 &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2 &operator=(CLTopKV2 &&) = default;
  /** Initialise the kernel's inputs and outputs.
   *
   * @note When locations of min and max occurrences are requested, the reported number of locations
   * is limited to the given array size.
   *
   * @param[in]  input     Input image. Data types supported: U8/S16/F32.
   * @param[in]  k         The value of `k`.
   * @param[out] values    Top k values. Data types supported: S32 if input type is U8/S16, F32 if
   * input type is F32.
   * @param[out] indices   indices related to top k values. Data types supported: S32 if input type
   * is U8/S16, F32 if input type is F32.
   */
  void configure(ICLTensor *input, int k, ICLTensor *values, ICLTensor *indices,
                 int total_bits = 32, int bits = 4);

  // Inherited methods overridden:
  void run() override;

private:
  void run_on_cpu();
  void run_on_gpu();
  void run_on_gpu_single_quicksort();

  uint32_t _k;
  uint32_t _total_bits;
  uint32_t _bits;
  uint32_t _radix;
  uint32_t _hist_buf_size;
  uint32_t _glob_sum_buf_size;
  uint32_t _n;

  ICLTensor *_input;
  ICLTensor *_values;
  ICLTensor *_indices;

  cl::Buffer _qs_idx_buf;
  cl::Buffer _qs_temp_buf;
  cl::Buffer _hist_buf;
  cl::Buffer _glob_sum_buf;
  cl::Buffer _temp_buf;
  cl::Buffer _first_negative_idx_buf;
  cl::Buffer _in_key_buf;
  cl::Buffer _out_key_buf;
  cl::Buffer _in_ind_buf;
  cl::Buffer _out_ind_buf;

  cl::Buffer *_p_in_key_buf;
  cl::Buffer *_p_out_key_buf;
  cl::Buffer *_p_in_ind_buf;
  cl::Buffer *_p_out_ind_buf;

  CLTopKV2Single _qs_kernel;
  CLTopKV2Init _init_kernel;
  CLRadixSortHistogram _hist_kernel;
  CLRadixSortScanHistogram _scan_hist_kernel;
  CLRadixSortGlobalScanHistogram _glob_scan_hist_kernel;
  CLRadixSortPasteHistogram _paste_hist_kernel;
  CLRadixSortReorder _reorder_kernel;
  CLTopKV2FindFirstNegative _find_first_negative_kernel;
  CLTopKV2ReorderNegatives _reorder_negatives_kernel;
  CLTopKV2Store _store_kernel;
};
}
#endif // __ARM_COMPUTE_CLTOPK_V2_H__
