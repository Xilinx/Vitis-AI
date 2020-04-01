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
#include <vitis/ai/yolov2.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vitis/ai/proto/dpu_model_param.pb.h>
namespace vitis {
namespace ai {
extern "C" proto::DpuModelParam *find(const std::string &model_name);
}
} // namespace vitis
using namespace std;
using namespace cv;

static std::vector<std::string> VOC_map{"aeroplane", "bicycle" , "bird", "boat", "bottle", "bus", "car",
                            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

int main(int argc, char *argv[]) {
    std::map<std::string, std::string> namemap {
        {"BASELINE", "yolov2_voc_baseline"},
        {"COMPRESS22G", "yolov2_voc_compress22G"},
        {"COMPRESS24G", "yolov2_voc_compress24G"},
        {"COMPRESS26G", "yolov2_voc_compress26G"},
    };
    if(argc != 3) {
        cerr << "usage: test_ssd_accuracy model_type image_path " << endl
             << "model_type is one of below  " << endl
             << "   BASELINE" << endl
             << "   COMPRESS22G" << endl
             << "   COMPRESS24G" << endl
             << "   COMPRESS26G" << endl;

        return -1;
    }
    string g_model_name = "none";
    g_model_name = argv[1];
    if (namemap.find(g_model_name) == namemap.end()) {
        cerr << "model_type is one of below  " << endl
             << "   BASELINE" << endl
             << "   COMPRESS22G" << endl
             << "   COMPRESS24G" << endl
             << "   COMPRESS26G" << endl
             <<  " it is " << g_model_name << endl;
        return -1;
    }

    cv::String path = argv[2];
    int length = path.size();
    vector<cv::String> files;
    cv::glob(path, files);

    auto model1 = vitis::ai::find(namemap[g_model_name]);
    model1->mutable_yolo_v3_param()->set_test_map(true);
    model1->mutable_yolo_v3_param()->set_conf_threshold(0.005);
    auto yolo = vitis::ai::YOLOv2::create(namemap[g_model_name], true);

    int count = files.size();
    cerr << "The image count = " << count << endl;
    for(int i = 0; i < count; i++){
        auto img = imread(files[i]);
        auto results = yolo->run(img);
        for(auto &box : results.bboxes){

            std::string label_name = VOC_map[box.label];
            //int label = box.label;
            float xmin = box.x * img.cols + 1;
            float ymin = box.y * img.rows + 1;
            float xmax = (box.x + box.width) * img.cols + 1;
            float ymax = (box.y + box.height) * img.rows + 1;
            if(xmin < 0) xmin = 1;
            if(ymin < 0) ymin = 1;
            if(xmax > img.cols) xmax = img.cols;
            if(ymax > img.rows) ymax = img.rows;
            float confidence = box.score;
            string imgname = String(files[i]).substr(length);
            std::cout << imgname.substr(0, 6) << " "
                << label_name << " " << confidence << " "
                << xmin << " " << ymin << " "
                << xmax << " " << ymax
                << std::endl;
        }
    }
    return 0;
}
