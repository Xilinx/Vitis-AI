/*
 * Copyright (C) 2020, Xilinx Inc - All rights reserved
 * Xilinx Runtime (XRT) Experimental APIs
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
/* header file for Runner APIs */
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vitis/ai/nnpp/tfssd.hpp>
#include <vitis/ai/nnpp/ssd.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#define _HW_SORT_EN_ 1

#if _HW_SORT_EN_
/* HW post proc-sort Init */
#include "tfssd/sort_xrt/sort_wrapper.h"
PPHandle* pphandle;
#endif

using namespace std;
using namespace cv;
using namespace std::chrono;

std::vector<const xir::Subgraph*> get_dpu_subgraph(
            const xir::Graph* graph) {
    auto root = graph->get_root_subgraph();
    auto children = root->children_topological_sort();
    auto ret = std::vector<const xir::Subgraph*>();
    for (auto c : children) {
        CHECK(c->has_attr("device"));
        auto device = c->get_attr<std::string>("device");
        if (device == "DPU") {
            ret.emplace_back(c);
        }
    }
    return ret;
}



int batchSize;
int modeScenario;
int num_channels_ = 3;
int is_server = 0;
int image_height_ = 300;
int image_width_ = 300;
std::thread   postThread;
string model = "ssd_mobilenet";
std::string dpuDir, imgDir;

std::string slurp(const char* filename);
vitis::ai::proto::DpuModelParam get_config() {
  
  string config_file;//
  if (model == "ssd_mobilenet")
    config_file = dpuDir+"/ssd_mobilenet_v1_coco_tf.prototxt";
  vitis::ai::proto::DpuModelParamList mlist;
  auto text = slurp(config_file.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);
  CHECK(ok) << "cannot parse config file. config_file=" << config_file;
  CHECK_EQ(mlist.model_size(), 1)
      << "only support one model per config file."
      << "config_file " << config_file << " "       //
      << "content: " << mlist.DebugString() << " "  //
      ;
  return mlist.model(0);
}
std::string slurp(const char* filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  CHECK(in.good()) << "failed to read config file. filename=" << filename;
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

void central_crop (const Mat& image, int height, int width, Mat& img) {
      int offset_h = (image.rows - height)/2;
      int offset_w = (image.cols - width)/2;
      Rect box(offset_w, offset_h, width, height);
      img = image(box);
}

void preProcessSsdmobilenet(Mat &orig, float scale, int width, int height, int8_t *data) {
        cv::Mat img, float_image;
        if (num_channels_ < 3) {
            cv::cvtColor(orig, float_image,  cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(orig, float_image,  cv::COLOR_BGR2RGB);
        }
 
        cv::resize((float_image), (img),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);
        int i = 0;
        for (int c=0; c < 3; c++) {
          for (int h=0; h < height; h++) {
            for (int w=0; w < width; w++) {
              data[0 + (3*h*width) + (w*3) + c] 
                = (int8_t)((img.at<Vec3b>(h,w)[c]*2/255.0-1.0)*scale);
            }
          }
        }
        i++;
    }

void load(float scale, int width, int height, int8_t *data)
{
    const int inSize = width * height * 3;
    if (model == "ssd_mobilenet") {
      image_height_ =image_width_ = 300;
    }
   
    string filename = "/root/imgdir//val2017/000000252219.jpg";        
     
    Mat orig = imread(filename);
    
    preProcessSsdmobilenet(orig, scale, width, height, data);

    std::cout << "Loaded samples success\n";
}

void runSSD(vart::Runner* runner)
{
	auto outTBuffs_ = dynamic_cast<vart::RunnerExt*>(runner)->get_outputs();
	auto inTBuffs_ = dynamic_cast<vart::RunnerExt*>(runner)->get_inputs();
	
	int8_t* output1 = (int8_t*)outTBuffs_[0]->data().first;
	int8_t* output2 = (int8_t*)outTBuffs_[1]->data().first;
	
	int8_t* input = (int8_t*)inTBuffs_[0]->data().first;
	
	vector<std::unique_ptr<vitis::ai::TFSSDPostProcess>> processor_;
	
	vitis::ai::proto::DpuModelParam config_ = get_config();
    
	processor_.push_back( vitis::ai::TFSSDPostProcess::create(
          image_width_, image_height_, 0.5, 0.125, config_));
		  
	float ip_scale = 64;
	int width_ = 300;
	int height_ = 300;
	load(ip_scale, width_, height_, input);
	
	auto ret = (runner)->execute_async(inTBuffs_, outTBuffs_);
   	(runner)->wait(uint32_t(ret.first), -1);

	auto results = processor_[0]->ssd_post_process(output1, output2);
     
	vector<float> resout;
	for (auto &box :  results[0].bboxes) {
        resout.push_back(float(box.y));
        resout.push_back(float(box.x));
        resout.push_back(float(box.y+box.height));
        resout.push_back(float(box.x+box.width));
        resout.push_back(float(box.score));
        resout.push_back(int(box.label));
	}
		
	return;
}

/* 
 * Usage: 
 * app.exe <options>
 */
int main(int argc, char **argv) {
  if(argc != 3){
   std::cerr << "invalid options <exe> <mode dir> <xmodel>\n";
   return -1; 
  }
  batchSize = 1;
  modeScenario = 0;
  int numSamples = -1;
  image_height_ =image_width_ = 300;
  dpuDir = argv[1]; 
  std::string xmodel_filename = argv[2]; 
 
  // runner
  std::unique_ptr<xir::Graph> graph0 = xir::Graph::deserialize(xmodel_filename);
  auto subgraph0 = graph0->get_root_subgraph();
  std::map<std::string, std::string> runset;
  runset.emplace("run","librt-engine.so");
  subgraph0->children_topological_sort()[1]->set_attr<std::string>("kernel", "DPUCAHX8L");
  subgraph0->children_topological_sort()[1]->set_attr("runner", runset);
  graph0->serialize("./dpu.xmodel");

  std::unique_ptr<xir::Graph> graph = xir::Graph::deserialize("./dpu.xmodel");
  auto subgraph = get_dpu_subgraph(graph.get());
 
  auto r = vart::Runner::create_runner(subgraph[0], "run");
  auto runner_ = std::move(r.get());
  
  runSSD(runner_);
  
  std::cout << "main return success\n"; 
  return 0;
}
