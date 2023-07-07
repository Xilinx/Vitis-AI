/*
 * Copyright 2019 Xilinx Inc.
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

#include <sys/stat.h>
#include "3DSegmentation_onnx.hpp"

using namespace ns_OnnxSegmentation3D;

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

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

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

template <typename T>
void writefilebin(string& filename, vector<T>& data) {
  ofstream output_file(filename, ios::binary);
  output_file.write(reinterpret_cast<char*>(data.data()),
                    sizeof(T) * data.size());
}

template <typename T>
void writefile(string& filename, vector<T>& data) {
  ofstream output_file(filename);
  for (size_t i = 0; i < data.size(); i++) output_file << (int)data[i] << endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <accuracy_file_list> <result_dir>" << std::endl;
    abort();
  }

  auto det = OnnxSegmentation3D::create(argv[1]);
  vector<string> names;
  LoadImageNames(argv[2], names);
  string out_dir = argv[3];

  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[3] << std::endl;
    return -1;
  }

  for (auto name : names) {
    vector<vector<vector<float>>> arrays(1, vector<vector<float>>(4));
    auto namesp = split(name, "-");
    auto path = namesp[0];
    auto outname = namesp[1];
    cout << path << endl;
    cout << outname << endl;
    string scan_x = path + "scan_x.txt";
    string scan_y = path + "scan_y.txt";
    string scan_z = path + "scan_z.txt";
    string remission = path + "scan_remission.txt";
    readfile(scan_x, arrays[0][0]);
    readfile(scan_y, arrays[0][1]);
    readfile(scan_z, arrays[0][2]);
    readfile(remission, arrays[0][3]);
    auto res = det->run(arrays);
    string result_bin_name = out_dir + "/" + outname + ".label";
    writefilebin(result_bin_name, res[0].array);
  }
  return 0;
}

