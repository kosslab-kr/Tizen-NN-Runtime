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

#ifndef __NNFW_UTIL_ENV_VAR__
#define __NNFW_UTIL_ENV_VAR__

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>


namespace nnfw
{
namespace util
{

class EnvVar
{
public:
  EnvVar(const std::string &key)
  {
    const char *value = std::getenv(key.c_str());
    if (value == nullptr)
    {
      // An empty string is considered as an empty value
      _value = "";
    }
    else
    {
      _value = value;
    }
  }

  std::string asString(const std::string &def) const
  {
    if (_value.empty())
      return def;
    return _value;
  }

  bool asBool(bool def) const
  {
    if (_value.empty())
      return def;
    static const std::array<std::string, 5> false_list{"0", "OFF", "FALSE", "N", "NO"};
    return std::find(false_list.begin(), false_list.end(), _value);
  }

  int asInt(int def) const
  {
    if (_value.empty())
      return def;
    return std::stoi(_value);
  }

private:
  std::string _value;
};

} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_ENV_VAR__
