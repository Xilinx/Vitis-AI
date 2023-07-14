/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <glog/logging.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "./cxxopts.hpp"

std::vector<float> read_from_bin_file(std::string file) {
  auto result_size = std::filesystem::file_size(file);
  std::vector<float> result(result_size / 4);

  CHECK(std::ifstream(file).read((char*)&result[0], result_size).good());
  return result;
}

int main(int argc, char* argv[]) {
  cxxopts::Options parser_def(argv[0], "\nCompare golden and dump data.\n");
  std::string golden_file = "";
  std::string dump_file = "";
  parser_def.add_options()  //
      ("g,gloden", "golden file.",
       cxxopts::value<std::string>(golden_file),  //
       "FILE")                                    //
      ("d,dump", "dump file",                     //
       cxxopts::value<std::string>(dump_file),    //
       "FILE")                                    //
      ("v,verbose", "prints various information useful for debugging.",
       cxxopts::value<bool>()->default_value("false"))  //
      ("p,print", "print diff",
       cxxopts::value<bool>()->default_value("false"))  //
      ("i,index", "print index float num",
       cxxopts::value<int>()->default_value("-1"))  //
      ("t,threshold",
       "threshold",                                    //
       cxxopts::value<float>()->default_value("0.5"),  //
       "input")                                        //
      ("h,help", "show help");
  cxxopts::ParseResult parse_results = parser_def.parse(argc, argv);
  if (parse_results.count("help")) {
    std::cout << parser_def.help();
    return 0;
  }
  bool verbose = parse_results["verbose"].as<bool>();
  bool print_diff = parse_results["print"].as<bool>();
  float threshold = parse_results["threshold"].as<float>();
  int index = parse_results["index"].as<int>();
  if (verbose) {
    std::cout << "threshold: " << threshold << "%" << std::endl;
  }

  if (golden_file == "") {
    std::cerr << "Please set golden file." << std::endl;
    return 1;
  }

  if (dump_file == "") {
    std::cerr << "Please set dump file." << std::endl;
    return 1;
  }

  auto golden_buf = read_from_bin_file(golden_file);
  auto dump_buf = read_from_bin_file(dump_file);
  if (verbose) {
    std::cout << "golden size: " << golden_buf.size()
              << ";\tdump size: " << dump_buf.size() << std::endl;
  }
  if (golden_buf.size() != dump_buf.size()) {
    std::cerr << "golden file size is not same as dump file." << std::endl;
    std::cerr << "golden size: " << golden_buf.size()
              << ";\tdump size: " << dump_buf.size() << std::endl;
  }

  if (index >= 0 && index < (int)golden_buf.size()) {
    std::cout << "golden : " << golden_buf[index]
              << ";\tdump : " << dump_buf[index] << std::endl;
    return 0;
  }

  bool is_diff = false;
  bool diff_is_more_than_threshold = false;
  for (auto i = 0u; i < golden_buf.size() / 4; ++i) {
    if (std::memcmp((void*)&golden_buf[i], (void*)&dump_buf[i],
                    sizeof(golden_buf[i])) != 0) {
      is_diff = true;
      if (verbose && print_diff) {
        std::cout << "index " << std::setw(8) << std::left << std::to_string(i)
                  << " diff.\tgolden: " << std::setw(10) << std::left
                  << golden_buf[i] << "\tdump: " << std::setw(10) << std::left
                  << dump_buf[i] << std::endl;
      }
    }

    if (std::abs((golden_buf[i] - dump_buf[i]) / golden_buf[i]) * 100.0f >
        threshold) {
      diff_is_more_than_threshold = true;
      if (verbose) {
        std::cout << "index " << std::setw(8) << std::left << std::to_string(i)
                  << " diff.\tgolden: " << std::setw(10) << std::left
                  << golden_buf[i] << "\tdump: " << std::setw(10) << std::left
                  << dump_buf[i] << "\tthreshold: " << threshold
                  << "%\t    diff_value: " << std::setw(10) << std::left
                  << std::abs(golden_buf[i] - dump_buf[i])
                  << "\t  diff_percert: "
                  << std::abs((golden_buf[i] - dump_buf[i]) / golden_buf[i]) *
                         100.0f
                  << "%" << std::endl;
      }
    }
  }

  if (is_diff) {
    if (!diff_is_more_than_threshold) {
      std::cerr
          << "Golden file and dump file are not same! But the diff is less "
             "than threshold."
          << std::endl;
    } else {
      std::cerr << "Golden file and dump file are diff." << std::endl;
    }
    return 1;
  }

  return 0;
}
