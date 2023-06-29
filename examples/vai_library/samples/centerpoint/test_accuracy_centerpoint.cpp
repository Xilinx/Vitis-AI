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

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/centerpoint.hpp>
#include <unistd.h>
#include <cassert>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;
using namespace cv;

template <class T>
static void readfile(string& filename, vector<T>& data) {
  ifstream input_file(filename, ios::binary);
  while (!input_file.eof()) {
    float orginData;
    input_file.read((char*)(&orginData), sizeof(float));
    data.push_back(orginData);
  }
  //cout << filename << " " << data.size() << endl;
}

/*
static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}
*/


void LoadFileNames(std::string const& filename,
                    std::vector<std::string>& files) {
  files.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    files.push_back(name);
  }

  fclose(fp);
}


int main(int argc, char* argv[]) {
  vector<string> names;
  if (argc < 6) {
    cout << "Please input 3 parameters into teminal.\n"
         << "First and second argu specify model, third is name list, fourth is bathpath, fifth is an output-boxes folder."
         << endl;
  }
  string model_0 = argv[1];
  string model_1 = argv[2];
  string bpath = argv[4];
  LoadFileNames(argv[3], names);
  string out_path = argv[5];
  auto centerpoint = vitis::ai::CenterPoint::create(
          model_0, model_1);
  if (!centerpoint) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  cout << argv[2] << " "  << names.size() << " files" << endl;
  for (auto& name : names) {
    ofstream out(out_path + "/" + name + ".txt");
    cout << out_path + "/" + name + ".txt" << endl;
    name = bpath + "/" + name + ".bin";
    auto input_file = name;
    cout << name << endl;
    std::vector<float> input;
    readfile(input_file, input);	
    for (auto k = 0 ;k < 20; k++) 
      cout << input[k] << " ";
    cout << endl;
    auto result = centerpoint->run(input);
    for (auto& i:result) {
      cout << "bbox:     ";
      for (auto& j:i.bbox) {
        cout << j << "    ";
        out << j << " ";
      }
      cout << "score:    " << i.score << endl; 
      out << i.score << " " << i.label << endl;
    }
  out.close();
  }
  return 0;
}
