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

#ifndef __TFLITE_RUN_ARGS_H__
#define __TFLITE_RUN_ARGS_H__

#include <string>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace TFLiteRun
{

class Args
{
public:
  Args(const int argc, char **argv);
  void print(void);

  const std::string &getTFLiteFilename(void) const { return _tflite_filename; }
  const std::string &getInputFilename(void) const { return _input_filename; }
  const std::string &getDumpFilename(void) const { return _dump_filename; }
  const std::string &getCompareFilename(void) const { return _compare_filename; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _tflite_filename;
  std::string _input_filename;
  std::string _dump_filename;
  std::string _compare_filename;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_ARGS_H__
