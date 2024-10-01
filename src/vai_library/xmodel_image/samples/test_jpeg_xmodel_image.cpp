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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <fstream>

#include "vitis/ai/xmodel_image.hpp"

#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

std::vector<std::string> g_image_files;
std::string g_xmodel_file;
std::string g_json_file;

static inline void Convert_Pb2Json(const google::protobuf::Message& message, 
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

static inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  auto usage = [=] {
    std::cout
        << argv[0] << "\n"
        << "-m <xmodel_file> : " << "\n"
	<< "\tset a xmodel file for testing" << "\n"
           "\tthere must be a py file with sample base file name in the same directory" << "\n"
	<< "-o <output_name> : " << "\n"
	<< "\tthe name of output json file" << "\n"
        << "-h : " << "\n"
	<< "\tfor help" << "\n"
        << std::endl;
  };

  while ((opt = getopt(argc, argv, "m:ho:")) != -1) {
    switch (opt) {
      case 'm':
        g_xmodel_file = optarg;
        break;
      case 'o':
	g_json_file = optarg;
	break;
      case 'h':
        usage();
        exit(0);
      default:
        std::cerr << "unknown arguments: " << opt << std::endl;
        usage();
        exit(1);
    }
  }

  for (auto i = optind; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  if (g_image_files.empty()) {
    std::cerr << "no input file" << std::endl;
    exit(1);
  }
  if (g_xmodel_file.empty()) {
    std::cerr << "no input model" << std::endl;
    exit(1);
  }
  return;
}

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
					size_t batchIndex,
                                        size_t batch) {
  std::vector<cv::Mat> images(batch);
  auto fileSize = files.size();
  for (auto index = 0u; index < batch; ++index) {
    auto fileIndex = index + batchIndex*batch;
    if (fileIndex == fileSize) {
      for (; index < batch; ++index) {
        images[index] = images[index-1];
      }
      break;
    }
    const auto& file = files[fileIndex];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
  return images;
}

static std::vector<vitis::ai::Mat> from_opencv(std::vector<cv::Mat>& images) {
  auto image_buffers = std::vector<vitis::ai::Mat>(images.size());
  for (auto i = 0u; i < image_buffers.size(); ++i) {
    image_buffers[i] =
        vitis::ai::Mat{images[i].rows, images[i].cols, images[i].type(),
                       images[i].data, images[i].step};
  }
  return image_buffers;
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  auto xmodel = vitis::ai::XmodelImage::create(g_xmodel_file);
  if (!xmodel) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto batchSize = xmodel->get_batch();
  auto fileSize = g_image_files.size();
  std::vector<vitis::ai::proto::DpuModelResult> results;
  if (batchSize == 0) return (-1);
  auto loopNum = (fileSize%batchSize)?(fileSize/batchSize+1):(fileSize/batchSize);
  for (auto i = 0u; i < loopNum; i++) {
    auto images = read_images(g_image_files, i, batchSize);
    auto image_buffers = from_opencv(images);
    auto result = xmodel->run(image_buffers);
    results.insert(results.end(), result.begin(), result.end());
  }

  int img_index = 0;
  for (const auto& r : results) {
    Convert_Pb2Json(r, img_index);
    img_index++;
  }

  return 0;
}
