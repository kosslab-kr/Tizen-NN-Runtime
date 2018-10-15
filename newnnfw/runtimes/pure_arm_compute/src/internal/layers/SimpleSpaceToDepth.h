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

#ifndef __SIMPLE_SPACE_TO_DEPTH_H__
#define __SIMPLE_SPACE_TO_DEPTH_H__

#include "internal/arm_compute.h"
#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/IFunction.h>

class SimpleSpaceToDepth : public ::arm_compute::IFunction
{
public:
  /** Initialise input and output
   *
   * @param[in]  input       First tensor input.
   * @param[out] output      Output tensor.
   * @param[in]  block_size  Block size.
   */
  void configure(::arm_compute::ITensor *input, ::arm_compute::ITensor *output, int32_t block_size,
                 const ::arm_compute::Coordinates &axises);

  void run() override;

private:
  ::arm_compute::ITensor *_input;
  ::arm_compute::ITensor *_output;
  int32_t _block_size;
  ::arm_compute::Coordinates _axises;
};

#endif /*__SIMPLE_SPACE_TO_DEPTH_H__ */
