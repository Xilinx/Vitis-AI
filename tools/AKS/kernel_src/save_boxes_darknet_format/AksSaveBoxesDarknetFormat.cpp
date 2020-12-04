/*
 * Copyright 2019 Xilinx Inc.
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
#include <iostream>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

// Box coordinates as (x, y, w, h)
struct XYWH {
  float x, y, w, h;
};

class SaveBoxesDarknetFormat : public AKS::KernelBase
{
  private:
    std::string _output_dir;
    ///. Save the output from postproc kernel in Darknet format for mAP calculation.
    XYWH _darknetStyleCoords(float img_w, float img_h, float llx, float lly, float urx, float ury);
  public:
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in, 
        std::vector<AKS::DataDescriptor*> &out, 
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings and 

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new SaveBoxesDarknetFormat();
  }

}//externC

void SaveBoxesDarknetFormat::nodeInit(AKS::NodeParams* nodeParams) {
  auto& tmp = nodeParams->_stringParams["output_dir"];
  _output_dir = tmp.empty() ? "." : tmp;
  if(_output_dir != "." && !boost::filesystem::exists(_output_dir)) {
    boost::filesystem::create_directory(_output_dir);
  }
}

int SaveBoxesDarknetFormat::exec_async (
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  // Get imgFile name from the full path
  const auto& imagePaths = dynParams->imagePaths;
  std::vector<int>& imgShapes = dynParams->_intVectorParams.at("img_dims");

  for(int b=0; b<imagePaths.size(); ++b) {
    std::vector<std::string> tokens;
    boost::split(tokens, imagePaths[b], boost::is_any_of("/,."));
    auto& imgFile = tokens[tokens.size()-2];

    // Append output_dir and .txt to get output file
    std::string output_file = _output_dir + "/" + imgFile + ".txt";
    ofstream f(output_file);
    if(!f) {
      std::cerr << "[WARNING] : Couldn't open " << output_file << std::endl;
      std::cerr << "[WARNING] : Check if path is correct" << std::endl;
      return -1;
    }

    // Get the boxes & image_dims from inputs and write it to file
    int img_h = imgShapes[b*3 + 1];
    int img_w = imgShapes[b*3 + 2];

    AKS::DataDescriptor* boxes = &(in[0]->data<AKS::DataDescriptor>()[b]);

    for(int i=0; i<boxes->getShape()[0]; ++i) {
      auto box     = boxes->data<float>() + i*6;
      int class_id = static_cast<int>(box[4]);
      float score  = box[5];
      float llx    = box[0];
      float lly    = box[1];
      float urx    = box[2];
      float ury    = box[3];

      XYWH xywh = _darknetStyleCoords(img_w, img_h, llx, lly, urx, ury);

      f << class_id << " " << score << " ";
      f << xywh.x << " " << xywh.y << " ";
      f << xywh.w << " " << xywh.h << '\n';
    }

    f.close();
  }
  return -1;
}

XYWH SaveBoxesDarknetFormat::_darknetStyleCoords(float img_w, float img_h, float llx, float lly, float urx, float ury)
{
  float dw = 1.0f/(img_w);
  float dh = 1.0f/(img_h);
  float x  = (llx + urx)/2.0f - 1.0f;
  float y  = (lly + ury)/2.0f - 1.0f;
  float w  = urx - llx;
  float h  = lly - ury;
  x = x*dw;
  w = w*dw;
  y = y*dh;
  h = h*dh;
  return XYWH{x,y,w,h};
}
