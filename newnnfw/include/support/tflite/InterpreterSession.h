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

#ifndef __NNFW_SUPPORT_TFLITE_INTERPRETER_SESSION_H__
#define __NNFW_SUPPORT_TFLITE_INTERPRETER_SESSION_H__

#include "Session.h"

namespace nnfw
{
namespace support
{
namespace tflite
{

class InterpreterSession final : public Session
{
public:
  InterpreterSession(::tflite::Interpreter *interp) : _interp{interp}
  {
    // DO NOTHING
  }

public:
  ::tflite::Interpreter *interp(void) override { return _interp; }

public:
  bool prepare(void) override
  {
    _interp->UseNNAPI(false);

    if (kTfLiteOk != _interp->AllocateTensors())
    {
      return false;
    }

    return true;
  }

  bool run(void) override
  {
    // Return true if Invoke returns kTfLiteOk
    return kTfLiteOk == _interp->Invoke();
  }

  bool teardown(void) override
  {
    // Do NOTHING currently
    return true;
  }

private:
  ::tflite::Interpreter * const _interp;
};

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_INTERPRETER_SESSION_H__
