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

#include <string.h>
#include <cstdlib>
#include <string>

#include "util/environment.h"

namespace nnfw
{
namespace util
{

int get_env_int(const char *name, int defaultValue)
{
  const char *value = std::getenv(name);
  if (value != nullptr)
    return std::stoi(value);
  return defaultValue;
}

bool get_env_bool(const char *name, bool defaultValue)
{
  const char *value = std::getenv(name);
  if (value != nullptr)
  {
    return std::stoi(value) != 0;
  }

  return defaultValue;
}

} // namespace util
} // namespace nnfw

namespace nnfw
{
namespace util
{
namespace env
{

IntAccessor::IntAccessor(const std::string &tag) : _tag{tag}
{
  // DO NOTHING
}

bool IntAccessor::access(int &out) const
{
  auto value = std::getenv(_tag.c_str());

  if (value == nullptr)
  {
    return false;
  }

  out = std::stoi(value);
  return true;
}

FloatAccessor::FloatAccessor(const std::string &tag) : _tag{tag}
{
  // DO NOTHING
}

bool FloatAccessor::access(float &out) const
{
  auto value = std::getenv(_tag.c_str());

  if (value == nullptr)
  {
    return false;
  }

  out = std::stof(value);
  return true;
}

} // namespace env
} // namespace util
} // namespace nnfw
