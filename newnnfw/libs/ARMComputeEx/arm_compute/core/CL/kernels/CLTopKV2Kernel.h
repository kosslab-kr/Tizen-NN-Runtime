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
#ifndef __ARM_COMPUTE_CLTOPKV2KERNEL_H__
#define __ARM_COMPUTE_CLTOPKV2KERNEL_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLKernel.h"

#include <array>

// these parameters can be changed
#define _ITEMS 16                          // number of items in a group
#define _GROUPS 4                          // the number of virtual processors is _ITEMS * _GROUPS
#define _HISTOSPLIT (_ITEMS * _GROUPS / 2) // number of splits of the histogram
#define PERMUT                             // store the final permutation
////////////////////////////////////////////////////////

namespace arm_compute
{
class ICLTensor;

class CLTopKV2Single : public ICLKernel
{
public:
  /** Constructor */
  CLTopKV2Single();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Single(const CLTopKV2Single &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Single &operator=(const CLTopKV2Single &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2Single(CLTopKV2Single &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2Single &operator=(CLTopKV2Single &&) = default;

  void configure(ICLTensor *input, ICLTensor *topk_values, ICLTensor *topk_indices,
                 cl::Buffer *indices, cl::Buffer *temp_stack, int k, int n);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_input;
  ICLTensor *_topk_values;
  ICLTensor *_topk_indices;
};

class CLTopKV2Init : public ICLKernel
{
public:
  /** Constructor */
  CLTopKV2Init();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Init(const CLTopKV2Init &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Init &operator=(const CLTopKV2Init &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2Init(CLTopKV2Init &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2Init &operator=(CLTopKV2Init &&) = default;

  void configure(ICLTensor *input, cl::Buffer *in_key_buf, cl::Buffer *in_ind_buf, int n);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_input;
};

class CLRadixSortHistogram : public ICLKernel
{
public:
  /** Constructor */
  CLRadixSortHistogram();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortHistogram(const CLRadixSortHistogram &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortHistogram &operator=(const CLRadixSortHistogram &) = delete;
  /** Allow instances of this class to be moved */
  CLRadixSortHistogram(CLRadixSortHistogram &&) = default;
  /** Allow instances of this class to be moved */
  CLRadixSortHistogram &operator=(CLRadixSortHistogram &&) = default;

  void configure(cl::Buffer *hist_buf, int bits, int n);

  void setPass(int pass, cl::Buffer *in_key_buf)
  {
    _pass = pass;
    _in_key_buf = in_key_buf;
  }

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  int _pass;
  cl::Buffer *_in_key_buf;
};

class CLRadixSortScanHistogram : public ICLKernel
{
public:
  /** Constructor */
  CLRadixSortScanHistogram();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortScanHistogram(const CLRadixSortScanHistogram &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortScanHistogram &operator=(const CLRadixSortScanHistogram &) = delete;
  /** Allow instances of this class to be moved */
  CLRadixSortScanHistogram(CLRadixSortScanHistogram &&) = default;
  /** Allow instances of this class to be moved */
  CLRadixSortScanHistogram &operator=(CLRadixSortScanHistogram &&) = default;

  void configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;
};

class CLRadixSortGlobalScanHistogram : public ICLKernel
{
public:
  /** Constructor */
  CLRadixSortGlobalScanHistogram();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortGlobalScanHistogram(const CLRadixSortGlobalScanHistogram &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortGlobalScanHistogram &operator=(const CLRadixSortGlobalScanHistogram &) = delete;
  /** Allow instances of this class to be moved */
  CLRadixSortGlobalScanHistogram(CLRadixSortGlobalScanHistogram &&) = default;
  /** Allow instances of this class to be moved */
  CLRadixSortGlobalScanHistogram &operator=(CLRadixSortGlobalScanHistogram &&) = default;

  void configure(cl::Buffer *glob_sum_buf, cl::Buffer *temp_buf, int bits);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;
};

class CLRadixSortPasteHistogram : public ICLKernel
{
public:
  /** Constructor */
  CLRadixSortPasteHistogram();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortPasteHistogram(const CLRadixSortPasteHistogram &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortPasteHistogram &operator=(const CLRadixSortPasteHistogram &) = delete;
  /** Allow instances of this class to be moved */
  CLRadixSortPasteHistogram(CLRadixSortPasteHistogram &&) = default;
  /** Allow instances of this class to be moved */
  CLRadixSortPasteHistogram &operator=(CLRadixSortPasteHistogram &&) = default;

  void configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;
};

class CLRadixSortReorder : public ICLKernel
{
public:
  /** Constructor */
  CLRadixSortReorder();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortReorder(const CLRadixSortReorder &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLRadixSortReorder &operator=(const CLRadixSortReorder &) = delete;
  /** Allow instances of this class to be moved */
  CLRadixSortReorder(CLRadixSortReorder &&) = default;
  /** Allow instances of this class to be moved */
  CLRadixSortReorder &operator=(CLRadixSortReorder &&) = default;

  void configure(cl::Buffer *hist_buf, int bits, int n);

  void setPass(int pass, cl::Buffer *in_key_buf, cl::Buffer *out_key_buf, cl::Buffer *in_ind_buf,
               cl::Buffer *out_ind_buf)
  {
    _pass = pass;
    _in_key_buf = in_key_buf;
    _out_key_buf = out_key_buf;
    _in_ind_buf = in_ind_buf;
    _out_ind_buf = out_ind_buf;
  }
  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  int _pass;
  cl::Buffer *_in_key_buf;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_in_ind_buf;
  cl::Buffer *_out_ind_buf;
};

class CLTopKV2FindFirstNegative : public ICLKernel
{
public:
  /** Constructor */
  CLTopKV2FindFirstNegative();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2FindFirstNegative(const CLTopKV2FindFirstNegative &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2FindFirstNegative &operator=(const CLTopKV2FindFirstNegative &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2FindFirstNegative(CLTopKV2FindFirstNegative &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2FindFirstNegative &operator=(CLTopKV2FindFirstNegative &&) = default;

  void configure(cl::Buffer *first_negative_idx_buf, int n);

  void setOutputBuffer(cl::Buffer *out_key_buf) { _out_key_buf = out_key_buf; }

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  cl::Buffer *_out_key_buf;
};

class CLTopKV2ReorderNegatives : public ICLKernel
{
public:
  /** Constructor */
  CLTopKV2ReorderNegatives();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2ReorderNegatives(const CLTopKV2ReorderNegatives &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2ReorderNegatives &operator=(const CLTopKV2ReorderNegatives &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2ReorderNegatives(CLTopKV2ReorderNegatives &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2ReorderNegatives &operator=(CLTopKV2ReorderNegatives &&) = default;

  void configure(cl::Buffer *first_negative_idx_buf, int n);

  void setBuffers(cl::Buffer *in_key_buf, cl::Buffer *out_key_buf, cl::Buffer *in_ind_buf,
                  cl::Buffer *out_ind_buf)
  {
    _in_key_buf = in_key_buf;
    _out_key_buf = out_key_buf;
    _in_ind_buf = in_ind_buf;
    _out_ind_buf = out_ind_buf;
  }

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  cl::Buffer *_in_key_buf;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_in_ind_buf;
  cl::Buffer *_out_ind_buf;
};

class CLTopKV2Store : public ICLKernel
{
public:
  /** Constructor */
  CLTopKV2Store();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Store(const CLTopKV2Store &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTopKV2Store &operator=(const CLTopKV2Store &) = delete;
  /** Allow instances of this class to be moved */
  CLTopKV2Store(CLTopKV2Store &&) = default;
  /** Allow instances of this class to be moved */
  CLTopKV2Store &operator=(CLTopKV2Store &&) = default;

  void configure(ICLTensor *values, ICLTensor *indices, int k, int n);

  void setOutputBuffers(cl::Buffer *out_key_buf, cl::Buffer *out_ind_buf);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_values;
  ICLTensor *_indices;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_out_ind_buf;
};

} // namespace arm_compute

#endif // __ARM_COMPUTE_CLTOPKV2KERNEL_H__
