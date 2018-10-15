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

#include "args.h"

#include <iostream>

#include <boost/filesystem.hpp>

namespace TFLiteRun
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{

  // General options
  po::options_description general("General options");

  // clang-format off
  general.add_options()
    ("help,h", "Display available options")
    ("input,i", po::value<std::string>(&_input_filename)->default_value(""), "Input filename")
    ("dump,d", po::value<std::string>()->default_value(""), "Output filename")
    ("compare,c", po::value<std::string>()->default_value(""), "filename to be compared with")
    ("tflite", po::value<std::string>()->required());
  // clang-format on

  _options.add(general);
  _positional.add("tflite", 1);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);
  po::notify(vm);

  {
    auto conflicting_options = [&](const std::string &o1, const std::string &o2) {
      if ((vm.count(o1) && !vm[o1].defaulted()) && (vm.count(o2) && !vm[o2].defaulted()))
      {
        throw boost::program_options::error(std::string("Two options '") + o1 + "' and '" + o2 +
                                            "' cannot be given at once.");
      }
    };

    conflicting_options("input", "compare");
  }

  if (vm.count("help"))
  {
    std::cout << "tflite_run\n\n";
    std::cout << "Usage: " << argv[0] << " <.tflite> [<options>]\n\n";
    std::cout << _options;
    std::cout << "\n";

    exit(0);
  }

  if (vm.count("input"))
  {
    _input_filename = vm["input"].as<std::string>();

    if (!_input_filename.empty())
    {
      if (!boost::filesystem::exists(_input_filename))
      {
        std::cerr << "input image file not found: " << _input_filename << "\n";
      }
    }
  }

  if (vm.count("dump"))
  {
    _dump_filename = vm["dump"].as<std::string>();
  }

  if (vm.count("compare"))
  {
    _compare_filename = vm["compare"].as<std::string>();
  }

  if (vm.count("tflite"))
  {
    _tflite_filename = vm["tflite"].as<std::string>();

    if (_tflite_filename.empty())
    {
      // TODO Print usage instead of the below message
      std::cerr << "Please specify tflite file. Run with `--help` for usage."
                << "\n";

      exit(1);
    }
    else
    {
      if (!boost::filesystem::exists(_tflite_filename))
      {
        std::cerr << "tflite file not found: " << _tflite_filename << "\n";
      }
    }
  }
}

} // end of namespace TFLiteRun
