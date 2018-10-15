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

#ifndef __NNFW_SUPPORT_TFLITE_NNAPI_SESSION_H__
#define __NNFW_SUPPORT_TFLITE_NNAPI_SESSION_H__

#include "Session.h"
#include "support/tflite/nnapi_delegate.h"

namespace nnfw
{
namespace support
{
namespace tflite
{

class NNAPISession final : public Session
{
public:
  NNAPISession(::tflite::Interpreter *interp) : _interp{interp}
  {
    // Construct Graph from Interpreter
    _delegate.BuildGraph(_interp);
  }

public:
  ::tflite::Interpreter *interp(void) override { return _interp; }

public:
  bool prepare(void) override
  {
    // Explicitly turn off T/F lite internal NNAPI delegation in order to use locally defined
    // NNAPI delegation.
    _interp->UseNNAPI(false);

    if (kTfLiteOk != _interp->AllocateTensors())
    {
      return false;
    }

    return true;
  }

  bool run(void) override
  {
    return kTfLiteOk == _delegate.Invoke(_interp);
  }

  bool teardown(void) override
  {
    // DO NOTHING
    return true;
  }

private:
  ::tflite::Interpreter * const _interp;
  nnfw::NNAPIDelegate _delegate;
};

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_NNAPI_SESSION_H__
