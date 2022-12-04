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
#include "textmountain_onnx.hpp"

std::vector<std::string> g_image_files;
void LoadListNames(const std::string& filename,  std::vector<std::string> &vlist)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if(!Tin)  {
     std::cout<<"Can't open the file " << filename << "\n";      exit(-1);
  }
  while( getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

static std::string getrealname(std::string& name) { return name.substr(0, name.find_last_of('.')); }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name>" << " <accuracy_file_list>  <test_dir> <result_dir>" << std::endl;
    abort(); 
  }
#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[1]));
#else
  auto model_name = std::string(argv[1]);
#endif
  LoadListNames( std::string(argv[2]), g_image_files);

  auto model = OnnxTextMountain::create(model_name); 
  auto ret = mkdir(argv[4], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[4] << std::endl;
    return -1;
  }

  ofstream Tout;

  std::vector<cv::Mat> imgs;
  for(int i=0; i<(int)g_image_files.size(); i++) {
    std::string matpath(argv[3]);
    matpath=matpath+"/"+g_image_files[i];
    imgs.clear();
    cv::Mat img = cv::imread( matpath );
    imgs.push_back(img);
    auto res = model->run(imgs);

    std::string txtname(argv[4]);
    txtname = txtname + "/res_" +  getrealname(g_image_files[i]) + ".txt" ;
    Tout.open(txtname, ios_base::out);
    if(!Tout) {
       cout<<"Can't open the file! " << txtname << "\n";
       return -1;
    }
    for(unsigned int j=0; j<res[0].res.size(); ++j) {
        Tout << res[0].res[j].box[0].x << "," << res[0].res[j].box[0].y <<","
             << res[0].res[j].box[1].x << "," << res[0].res[j].box[1].y <<","
             << res[0].res[j].box[2].x << "," << res[0].res[j].box[2].y <<","
             << res[0].res[j].box[3].x << "," << res[0].res[j].box[3].y <<","
             << res[0].res[j].score <<"\n";
    }
    Tout.close();
  }
  return 0;
}




