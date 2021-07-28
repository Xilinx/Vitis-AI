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

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std::chrono;

#define DISPLAY_ID 0


#include <CL/cl.h>
#include "xcl2.hpp"
class PPHandle {
public:

  cl::Context contxt;
  cl::Device device;
  cl::Kernel kernel;
  cl::CommandQueue q;
  cl::Buffer paramsbuf;

};
#include "hw_preproc.h"
PPHandle* handle;

using namespace std;
using namespace cv;

GraphInfo shapes;

const string baseImagePath = "../images/";
const string wordsPath = "./";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const& path, vector<string>& images) {
  images.clear();
  struct dirent* entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const& path, vector<string>& kinds) {
  kinds.clear();
  ifstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float* data, size_t size, float* result) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i]);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float* d, int size, int k, vector<string>& vkinds) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
           vkinds[ki.second].c_str());
    q.pop();
  }
}

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runResnet50(vart::Runner* runner, bool hw_pp) {
  /* Mean value for ResNet50 specified in Caffe prototxt */
  vector<string> kinds, images;

  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* Load all kinds words.*/
  LoadWords(wordsPath + "words.txt", kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: No words exist in file words.txt." << endl;
    return;
  }
  float mean[3] = {104, 107, 123};

  /* get in/out tensors and dims*/
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  /*get shape info*/
  int outSize = shapes.outTensorList[0].size;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;

  int batchSize = in_dims[0];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  vector<Mat> imageList;
  float* imageInputs = new float[inSize * batchSize];

  float* softmax = new float[outSize];
  float* FCResult = new float[batchSize * outSize];

  long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0;


  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
  if(hw_pp)
  pp_kernel_init(handle,"/media/sd-mmcblk0p1/dpu.xclbin", "pp_pipeline_accel",0, mean);

  auto start_prof = std::chrono::system_clock::now();

  std::cout << "number of images being run: " << images.size() << "\n";
    int count = images.size();
  if(hw_pp)  
	std::cout << "Running with HW Accel Pre-process"<<std::endl;
	else
  std::cout << "Running with SW Pre-process"<<std::endl;
  /*run with batch*/
  for (unsigned int n = 0; n < images.size(); n += batchSize) {
    unsigned int runSize =
        (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
    in_dims[0] = runSize;
    out_dims[0] = batchSize;

    

    for (unsigned int i = 0; i < runSize; i++) {

      auto t1 = std::chrono::system_clock::now();
      Mat image = imread(baseImagePath + images[n + i]);
      auto t2 = std::chrono::system_clock::now();
			auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
			imread_time += value_t1.count();
			
			auto pre_t1 = std::chrono::system_clock::now();


      /*image pre-process*/
      if(hw_pp)
      preprocess(handle, image, inHeight, inWidth, mean, imageInputs + i * inSize);
      else{
      Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
      resize(image, image2, Size(inHeight, inWidth), 0, 0);
      for (int h = 0; h < inHeight; h++) {
        for (int w = 0; w < inWidth; w++) {
          for (int c = 0; c < 3; c++) {
            imageInputs[i * inSize + h * inWidth * 3 + w * 3 + c] =
                image2.at<Vec3b>(h, w)[c] - mean[c];
          }
        }
      }
    }

      auto pre_t2 = std::chrono::system_clock::now();
			auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2-pre_t1);
			pre_time += prevalue_t1.count();
      imageList.push_back(image);
    }

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(
        inputTensors[0]->get_name(), in_dims,
        xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(
        outputTensors[0]->get_name(), out_dims,
        xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    /*run*/
    auto exec_t1 = std::chrono::system_clock::now();
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    auto exec_t2 = std::chrono::system_clock::now();
		auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2-exec_t1);
		exec_time += execvalue_t1.count();

    for (unsigned int i = 0; i < runSize; i++) {
      #if DISPLAY_ID
      cout << "\nImage : " << images[n + i] << endl;
      #endif
      /* Calculate softmax on CPU and display TOP-5 classification results */
      CPUCalcSoftmax(&FCResult[i * outSize], outSize, softmax);
      #if DISPLAY_ID
      TopK(softmax, outSize, 5, kinds);
      #endif
      /* Display the impage */
     /* bool quiet = (getenv("QUIET_RUN") != nullptr);
      if (!quiet){
        cv::imshow("Classification of ResNet50", imageList[i]);
        cv::waitKey(10000);
      }*/
    }
    imageList.clear();
    inputs.clear();
    outputs.clear();
  }

 
  auto end_prof = std::chrono::system_clock::now();
	auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(end_prof- start_prof);
	long e2e_time = value_t1.count();

    std::cout << "E2E Performance: " << 1000000.0 / ((float)(e2e_time/count)) << " fps\n";
		std::cout << "imread latency: " << (float)(imread_time/count) / 1000 << "ms\n";
		std::cout << "pre latency: " << (float)(pre_time/count) / 1000 << "ms\n";
    std::cout << "exec latency: " << (float)(exec_time/count) / 1000 << "ms\n";
  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char* argv[]) {
  // Check args
  if (argc != 3) {
    cout << "Usage of resnet50 demo: ./app_resnet_versal_waa [model_file]  [0 or 1 use 0 for running with software pre-process and 1 for hardware accelerated pre-process]" << endl;
    return -1;
  }
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  bool hw_pp = atoi(argv[2]);
  CHECK_EQ(subgraph.size(), 1u)
      << "resnet50 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  /*create runner*/
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  /*run with batch*/
  runResnet50(runner.get(), hw_pp);
  return 0;
}
