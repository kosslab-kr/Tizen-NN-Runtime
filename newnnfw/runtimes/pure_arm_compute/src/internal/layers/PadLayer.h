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

#ifndef __PAD_LAYER_H__
#define __PAD_LAYER_H__

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/functions/CLFillBorder.h>

class PadLayer : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ICLTensor *input, ::arm_compute::ICLTensor *output,
                 unsigned int border_width);
  void run(void) override;

private:
  ::arm_compute::ICLTensor *_input;
  ::arm_compute::ICLTensor *_output;
  int _border_width;
  int _output_height;
  int _output_width;

  ::arm_compute::CLFillBorder _fillborderkernel;
  void populateOutput();
};

#endif // __PAD_LAYER_H__
