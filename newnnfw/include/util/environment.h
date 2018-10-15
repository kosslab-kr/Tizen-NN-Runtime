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

#ifndef __UTIL_ENVIRONMENT_H__
#define __UTIL_ENVIRONMENT_H__

namespace nnfw
{
namespace util
{

int get_env_int(const char *name, int defaultValue = 0);
bool get_env_bool(const char *name, bool defaultValue = false);
}
}

#include <string>

namespace nnfw
{
namespace util
{
namespace env
{

template <typename T> struct Accessor
{
  virtual ~Accessor() = default;

  virtual bool access(T &out) const = 0;
};

class IntAccessor : public Accessor<int>
{
public:
  IntAccessor(const std::string &tag);

public:
  bool access(int &out) const override;

private:
  std::string _tag;
};

class FloatAccessor : public Accessor<float>
{
public:
  FloatAccessor(const std::string &tag);

public:
  bool access(float &out) const override;

private:
  std::string _tag;
};

} // namespace env
} // namespace util
} // namespace nnfw

#endif // __UTIL_ENVIRONMENT_H__
