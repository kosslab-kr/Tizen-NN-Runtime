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
#ifndef __ARM_COMPUTE_CLSTRIDEDSLICEKERNEL_H__
#define __ARM_COMPUTE_CLSTRIDEDSLICEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to extract a strided slice of a tensor */
class CLStridedSliceKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLStridedSliceKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLStridedSliceKernel(const CLStridedSliceKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLStridedSliceKernel &operator=(const CLStridedSliceKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLStridedSliceKernel(CLStridedSliceKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLStridedSliceKernel &operator=(CLStridedSliceKernel &&) = default;
  /** Default destructor */
  ~CLStridedSliceKernel() = default;
  /** Set the input and output of the kernel
   *
   * @param[in]  input          Source tensor. Data type supported:
   * U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/F16/F32
   * @param[out] output         Destination tensor. Data type supported: Same as @p input
   * @param[in]  beginData      The begin tensor. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  endData        The end tensor. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  strideData     The stride tensor. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  beginMask      Mask for begin
   * @param[in]  endMask        Mask for end
   * @param[in]  shrinkAxisMask Mask for shrink axis.
   *
   */
  void configure(const ICLTensor *input, ICLTensor *output, ICLTensor *beginData,
                 ICLTensor *endData, ICLTensor *stridesData, int32_t beginMask, int32_t endMask,
                 int32_t shrinkAxisMask);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLStridedSliceKernel
   *
   * @param[in]  input          The input tensor info. Data types supported:
   * U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/F16/F32
   * @param[in]  output         The output tensor info, Data types supported: same as @p input1.
   * @param[in]  begin          The begin tensor info. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  end            The end tensor info. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  stride         The stride tensor info. Data types supported: S32.
   *                            The number of dimensions must be 1.
   *                            The length must be the same as the number of dimensions of input.
   * @param[in]  beginMask      Mask for begin
   * @param[in]  endMask        Mask for end
   * @param[in]  shrinkAxisMask Mask for shrink axis.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *begin, const ITensorInfo *end,
                         const ITensorInfo *stride, int32_t beginMask, int32_t endMask,
                         int32_t shrinkAxisMask);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input; /** Source tensor */
  ICLTensor *_output;      /** Destination tensor */
  ICLTensor *_beginData;   /** Start indices of input tensor */
  ICLTensor *_endData;     /** Stop indices of input tensor */
  ICLTensor *_stridesData; /** Strides tensor */
  int32_t _beginMask;      /** Begin mask */
  int32_t _endMask;        /** End mask */
  int32_t _shrinkAxisMask; /** Shrink axis mask */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLSTRIDEDSLICEKERNEL_H__ */
