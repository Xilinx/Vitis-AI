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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

#include "./cxxopts.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  cxxopts::Options parser_def(argv[0], "show image");
  std::string file_name = "";
  int width = 0;
  int height = 0;
  int top = 0;
  int left = 0;
  int bottom = 0;
  int right = 0;
  int num_of_channels = 3;
  int channel = 0;
  int elt_size = 1;
  parser_def.add_options()  //
      ("f,file", "binary file name.",
       cxxopts::value<std::string>(file_name),    //
       "FILE")                                    //
      ("w,width", "width",                        //
       cxxopts::value<int>(width),                //
       "INT")                                     //
      ("h,height", "height",                      //
       cxxopts::value<int>(height),               //
       "INT")                                     //
      ("t,top", "top",                            //
       cxxopts::value<int>(top),                  //
       "INT")                                     //
      ("b,bottom", "bottom",                      //
       cxxopts::value<int>(bottom),               //
       "INT")                                     //
      ("l,left", "left",                          //
       cxxopts::value<int>(left),                 //
       "INT")                                     //
      ("r,right", "right",                        //
       cxxopts::value<int>(right),                //
       "INT")                                     //
      ("num_of_channels", "number of channels",   //
       cxxopts::value<int>(num_of_channels),      //
       "INT")                                     //
      ("c,channel", "channel",                    //
       cxxopts::value<int>(channel),              //
       "INT")                                     //
      ("e,element-size", "element size 1,2,4,8",  //
       cxxopts::value<int>(elt_size),             //
       "INT")("help", "show help");
  cxxopts::ParseResult parse_results = parser_def.parse(argc, argv);
  if ((argc == 0) || parse_results.count("help")) {
    std::cout << parser_def.help();
    return 0;
  }
  if (bottom == 0) {
    bottom = height;
  }
  if (right == 0) {
    right = width;
  }
  uint64_t mask = -1;
  mask = mask << (elt_size * 8);
  mask = ~mask;
  cout << "width " << width << " "                      //
       << "height " << height << " "                    //
       << "left " << left << " "                        //
       << "right " << right << " "                      //
       << "top " << top << " "                          //
       << "bottom " << bottom << " "                    //
       << "num_of_channels " << num_of_channels << " "  //
       << "elt_size " << elt_size << " "                //
       << std::hex << "0x" << mask << " " << std::dec << endl;
  ;
  auto buf = vector<char>();
  buf.resize((size_t)width * height * num_of_channels * elt_size);
  CHECK(std::ifstream(file_name).read((char*)&buf[0], buf.size()).good())
      << "failed to read baseline from " << file_name;
  for (auto y = top; y < bottom; ++y) {
    for (auto x = left; x < right; ++x) {
      auto index = y * width * num_of_channels + x * num_of_channels + channel;
      unsigned int value = buf[index * elt_size] & mask;
      cout << ' ' << std::hex << std::setfill('0') << std::setw(elt_size * 2)
           << (unsigned int)value;
    }
    cout << "\n";
  }
  return 0;
}
