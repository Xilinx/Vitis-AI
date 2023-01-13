/*
 * Copyright 2019 xilinx Inc.
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
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

#include "vitis/ai/general.hpp"

std::string g_model_name = "resnet50";
std::string g_image_file = "";
std::string g_json_file = "";

static void usage() {
  std::cout << "usage: test_general <model_name> <img_file> [output_json_file]" << std::endl;
}

static void Convert_Pb2Json(const google::protobuf::Message& message,
		                const int msg_index) {
  std::string jsonStr;
  google::protobuf::util::JsonOptions options;

  google::protobuf::util::MessageToJsonString(message, &jsonStr, options);

  if(g_json_file.empty()) {
    std::cout << jsonStr << std::endl;
  } else {
    std::string outFile = g_json_file;
    std::string msgStr = std::to_string(msg_index);
 
    if (std::string::npos != outFile.find_last_of(".")) {
      outFile.insert(outFile.find_last_of("."), std::string("_batch_").append(msgStr));
    } else {
      outFile.append("_batch_").append(msgStr);
    }
    CHECK(std::ofstream(outFile).write(jsonStr.append("\n").c_str(), jsonStr.size()).good())
	    << " failed to write to " << outFile;
  }
  return;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    usage();
    return 1;
  }
  g_model_name = argv[1];
  g_image_file = argv[2];
  if (argc == 4) g_json_file = argv[3];

  auto image = cv::imread(g_image_file);
  if (image.empty()) {
    std::cerr << "cannot load " << g_image_file << std::endl;
    abort();
  }
  auto model = vitis::ai::General::create(g_model_name, true);
  if (model) {
    auto result = model->run(image);
    std::cerr << "result = " << result.DebugString() << std::endl;
    Convert_Pb2Json(result, 0);
  } else {
    std::cerr << "no such model, ls -l /usr/share/vitis-ai-library to see available "
            "models"
         << std::endl;
  }
  return 0;
}
