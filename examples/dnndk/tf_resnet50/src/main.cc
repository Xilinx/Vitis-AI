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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
/* header files for Vitis AI advanced APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

#define KRENEL_CONV "tf_resnet50_0"

#define TASK_CONV_INPUT "resnet_v1_50_conv1_Conv2D"
#define TASK_CONV_OUTPUT "resnet_v1_50_logits_Conv2D"

const string baseImagePath = "../dataset/image500_640_480/";

/*List all images's name in path.*/
void ListImages(std::string const &path, std::vector<std::string> &images) {
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "bmp") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/*Load all kinds*/
void LoadWords(std::string const &path, std::vector<std::string> &kinds) {
  kinds.clear();
  std::fstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  std::string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
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
void TopK(const float *d, int size, int k, std::vector<std::string> &vkind) {
  assert(d && size > 0 && k > 0);
  std::priority_queue<std::pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(std::pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    std::pair<float, int> ki = q.top();
	/* Note: For current tensorflow Resnet model, there are 1001 kinds.*/
	int real_ki = ki.second;
    fprintf(stdout, "top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
            vkind[real_ki].c_str());
    q.pop();
  }
}

void central_crop(const Mat& image, int height, int width, Mat& img) {
  int offset_h = (image.rows - height)/2;
  int offset_w = (image.cols - width)/2;
  Rect box(offset_w, offset_h, width, height);
  img = image(box);
}


void change_bgr(const Mat& image, int8_t* data, float scale, float* mean) {
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < image.rows; ++j)
      for(int k = 0; k < image.cols; ++k) {
		    data[j*image.rows*3+k*3+2-i] = (image.at<Vec3b>(j,k)[i] - (int8_t)mean[i]) * scale;
      }

}

/**
 * @brief set input image
 *
 * @param task - pointer to Resnet50 CONV Task
 * @param input_node - input node of Resnet50
 * @param image - the input image
 * @param  mean - mean of Resnet50
 *
 * @return none
 */
inline void set_input_image(DPUTask *task, const string& input_node, const cv::Mat& image, float* mean){
  Mat cropped_img;
  DPUTensor* dpu_in = dpuGetInputTensor(task, input_node.c_str());
  float scale = dpuGetTensorScale(dpu_in);
  int width = dpuGetTensorWidth(dpu_in);
  int height = dpuGetTensorHeight(dpu_in);
  int size = dpuGetTensorSize(dpu_in);
  vector<int8_t> abc(size);
  central_crop(image, height, width, cropped_img);

  int8_t* data = dpuGetTensorAddress(dpu_in);
  change_bgr(cropped_img, data, scale, mean);
}

/**
 * @brief Run DPU CONV Task and FC Task for Resnet50
 *
 * @param taskConv - pointer to Resnet50 CONV Task
 * @param taskFC - pointer to Resnet50 FC Task
 *
 * @return none
 */
void runResnet(DPUTask *taskConv) {
  assert(taskConv);

  vector<string> kinds, images;

  /*Load all image names */
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    fprintf(stdout, "[debug] %s is the directory\n", baseImagePath.c_str());
    cerr << "\nError: Not images exist in " << baseImagePath << endl;
    return;
  }

  /*Load all kinds words.*/
  LoadWords(baseImagePath + "words.txt", kinds);

  /* Get the output Tensor for Resnet50 Task  */
  int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskConv, TASK_CONV_OUTPUT);
  /* Get size of the output Tensor for Resnet50 Task  */
  int size = dpuGetOutputTensorSize(taskConv, TASK_CONV_OUTPUT);
  /* Get channel count of the output Tensor for FC Task  */
  int channel = dpuGetOutputTensorChannel(taskConv, TASK_CONV_OUTPUT);
  /* Get scale of the output Tensor for Resnet50 Task  */
  float out_scale = dpuGetOutputTensorScale(taskConv, TASK_CONV_OUTPUT);

  float *softmax = new float[size];

  for (auto &image_name : images) {
    cout << "\nLoad image : " << image_name << endl;
    Mat image = imread(baseImagePath + image_name);

    /* Set image into Conv Task with mean value   */
    float mean[3] = {0, 0, 0};

    set_input_image(taskConv, TASK_CONV_INPUT, image, mean);

    /* Run Resnet50 CONV part */
    cout << "\nRun Resnet50 CONV ..." << endl;
    dpuRunTask(taskConv);

    /* Calculate softmax on CPU and show TOP5 classification result */
    dpuRunSoftmax(outAddr, softmax, channel, size/channel, out_scale);
    TopK(softmax, channel, 5, kinds);

    /* Show the image */
    cv::imshow("Image", image);
    cv::waitKey(1);
  }
  delete[] softmax;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Vitis AI advanced APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char *argv[]) {
  /* DPU Kernels/Tasks for runing Resnet50 */
  DPUKernel *kernelConv;
  DPUTask *taskConv;

  /* Attach to DPU driver and prepare for runing */
  dpuOpen();

  /* Create DPU Kernels for CONV & FC Nodes in Resnet50 */
  kernelConv = dpuLoadKernel(KRENEL_CONV);

  /* Create DPU Tasks for CONV & FC Nodes in Resnet50 */
  taskConv = dpuCreateTask(kernelConv, 0);

  /* Run CONV & FC Kernels for Resnet50 */
  runResnet(taskConv);

  /* Destroy DPU Tasks & free resources */
  dpuDestroyTask(taskConv);

  /* Destroy DPU Kernels & free resources */
  dpuDestroyKernel(kernelConv);

  /* Dettach from DPU driver & release resources */
  dpuClose();

  return 0;
}
