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
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/3Dsegmentation.hpp>
#include <vitis/ai/profiling.hpp>
#include <fstream>
#include <thread>
using namespace std;
using namespace vitis::ai;
vector<string> names;

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
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

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}

void readfile(string& filename, vector<float>& data) {
  ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    istringstream ss(line);
    float num;
    ss >> num;
    data.push_back(num);
  }
  cout << filename << " " << data.size() << endl;
}

template<typename T> 
void writefilebin(string& filename, vector<T>& data) {
  ofstream output_file(filename, ios::binary);
  output_file.write(reinterpret_cast<char *>(data.data()), sizeof(T) * data.size());
}


template<typename T> 
void writefile(string& filename, vector<T>& data) {
  ofstream output_file(filename);
  for(size_t i = 0; i < data.size(); i++) 
    output_file << (int)data[i] << endl;
}

void accuracy_thread(Segmentation3D* seg, int i, int t_n){

  for(int j=i; j<(int)names.size(); j+=t_n){
    std::string name = names[j];
    vector<vector<float>> arrays(4);
    auto namesp = split(name, "-");
    auto path = namesp[0];
    auto outname = namesp[1];
    // cout << path << endl;
    // cout << outname << endl;
    string scan_x = path + "scan_x.txt";
    string scan_y = path + "scan_y.txt";
    string scan_z = path + "scan_z.txt";
    string remission = path + "scan_remission.txt";
    readfile(scan_x, arrays[0]);
    readfile(scan_y, arrays[1]);
    readfile(scan_z, arrays[2]);
    readfile(remission, arrays[3]);
    auto res = seg->run(arrays);
    string result_bin_name = "result/" + outname + ".label";
    writefilebin(result_bin_name, res.array);
  }
}

int main(int argc, char *argv[]) {
  // bool preprocess = !(getenv("PRE") != nullptr);
  if (argc<2) {
      std::cout <<" usage: " << argv[0] << " <model> [ thread_num ] \n";
      return 0;
  }

  LoadImageNames(argv[2], names);

  int t_n=2;
  if (argc==2 ) t_n = atoi(argv[2]);
  t_n = std::min(100, std::max(0, t_n));

  std::vector<std::thread> vth;
  std::vector< std::unique_ptr<Segmentation3D>> vseg;
  for(int i=0; i<t_n; i++) {
    auto seg = vitis::ai::Segmentation3D::create(argv[1]);
    if (!seg) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
    }  

    vseg.emplace_back(std::move(seg));
    vth.emplace_back( std::thread( &accuracy_thread, vseg[i].get(), i , t_n));
  }
  for(int i=0; i<t_n; i++) {
    vth[i].join();
  }

  return 0;
}
