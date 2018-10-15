/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSTRIDEDSLICE_H__
#define __ARM_COMPUTE_CLSTRIDEDSLICE_H__

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLStridedSliceKernel */
class CLStridedSlice : public ICLSimpleFunction
{
public:
  /** Initialise the kernel's inputs and outputs
   *
   * @param[in]  input  First tensor input. Data type supported:
   * U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/F16/F32
   * @param[out] output Output tensor. Data type supported: Same as @p input
   */
  void configure(const ICLTensor *input, ICLTensor *output, ICLTensor *beginData,
                 ICLTensor *endData, ICLTensor *stridesData, int32_t beginMask, int32_t endMask,
                 int32_t shrinkAxisMask);
};

class CLStridedSliceCPU : public IFunction
{
public:
  /** Initialise inputs and outputs
   *
   * @param[in]  input  First tensor input.
   * @param[out] output Output tensor.
   */
  void configure(ICLTensor *input, ICLTensor *output, ICLTensor *beginData, ICLTensor *endData,
                 ICLTensor *stridesData, int32_t beginMask, int32_t endMask,
                 int32_t shrinkAxisMask);

  void run() override;

private:
  void run_on_cpu();

  ICLTensor *_input;
  ICLTensor *_output;
  ICLTensor *_beginData;
  ICLTensor *_endData;
  ICLTensor *_stridesData;
  int32_t _beginMask;
  int32_t _endMask;
  int32_t _shrinkAxisMask;
};
}
#endif /*__ARM_COMPUTE_CLSTRIDEDSLICE_H__ */
