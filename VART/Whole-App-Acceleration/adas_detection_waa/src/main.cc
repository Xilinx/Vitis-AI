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
#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include "common.h"

#include "utils.h"
#include <opencv2/opencv.hpp>

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

using namespace std;
using namespace cv;
using namespace std::chrono;

#define NMS_THRESHOLD 0.3f
bool usePPFlag = true;
int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reding input frame
chrono::system_clock::time_point start_time;

typedef pair<int, Mat> imagePair;
class paircomp {
    public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

const string baseImagePath = "./data/";
/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
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

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protection of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;
GraphInfo shapes;
/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
int runTotal;
PPHandle* handle;
void setInputImageForYOLO(vart::Runner* runner, float *data, const Mat& frame, float* mean) {
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;
  int size = shapes.inTensorList[0].size;
  if(usePPFlag){
    /* pre-processing for input frame */
    preprocess(handle, frame, height, width, data);
    float scalei = pow(2, 7);
    for(int i = 0; i < size; ++i) {
      if(data[i] < 0) data[i] = (float)(127/scalei);
    }
  } else {
    Mat img_copy;
    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);
    vector<float> bb(size);
    if (runner->get_tensor_format() == vart::Runner::TensorFormat::NHWC) {
      for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
          for(int a = 0; a < 3; ++a) {
            bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
          }
        }
      }
    } else {
      for(int a = 0; a < 3; ++a) {
        for(int b = 0; b < height; ++b) {
          for(int c = 0; c < width; ++c) {
            bb[a*width*height + b*width + c] = img_yolo.data[a*height*width + b*width + c];
          }
        }
      }
    }
    float scale = pow(2, 7);
    for(int i = 0; i < size; ++i) {
      data[i] = bb.data()[i];
      if(data[i] < 0) data[i] = (float)(127/scale);
    }
    free_image(img_new);
    free_image(img_yolo);
  }
}

/**
 * @brief Thread entry for reading image frame from the input video file
 *
 * @param fileName - pointer to video file name
 *
 * @return none
 */
void readFrame(vector<string> images){
  start_time = chrono::system_clock::now();

  while(true){ 
    if(idxInputImage< images.size()) 
    {
      Mat img;
      img = imread(baseImagePath + images[idxInputImage]);
      mtxQueueInput.lock();
      queueInput.push(make_pair(idxInputImage++, img));
      mtxQueueInput.unlock();
    }
  }

  exit(0);
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame(vector<string> images) {
  Mat frame;
  while (true) {
    //std::cout << "*******display" << std::endl;
    mtxQueueShow.lock();

    if (queueShow.empty()) {
      mtxQueueShow.unlock();
    } 
    else if (idxShowImage == queueShow.top().first) {
      frame = queueShow.top().second;
      cv::imwrite("./output/" + images[idxShowImage], frame);
      idxShowImage++;
      queueShow.pop();
      mtxQueueShow.unlock();
      if(idxShowImage==runTotal) {
        bReading = false;
        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        stringstream buffer_str;
        buffer_str << fixed << setprecision(1)
          << (float)runTotal / (dura / 1000000.f);
        string a = buffer_str.str() + " FPS";
        std::cout << "Performance:" << a << std::endl;

        exit(0);
      }
    } else {
      mtxQueueShow.unlock();
    }
  }
}

/**
 * @brief Post process after the running of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(vart::Runner* runner, Mat& frame,vector<float*> results, int sWidth, int sHeight){
  const string classes[3] = {"car", "person", "cycle"};

  /* four output nodes of YOLO-v3 */
  // const string outputs_node[4] = {"layer81_conv", "layer93_conv",
  //                                    "layer105_conv", "layer117_conv"};
  vector<vector<float>> boxes;
  //auto  outputTensors = runner->get_output_tensors();
  for(int ii = 0; ii < 4; ii++) {
    int width = shapes.outTensorList[ii].width;
    int height = shapes.outTensorList[ii].height;
    int channel = shapes.outTensorList[ii].channel;
    int sizeOut = channel * width * height;
    vector<float> result(sizeOut);
    boxes.reserve(sizeOut);

    /* Store every output node results */
    get_output(results[ii], sizeOut,  channel, height, width, result);

    /* Store the object detection frames as coordinate information  */
    detect(boxes, result, channel, height, width, ii, sHeight, sWidth);
  }

  /* Restore the correct coordinate frame of the original image */
  correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

  /* Apply the computation for NMS */
  vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

  float h = frame.rows;
  float w = frame.cols;
  for(size_t i = 0; i < res.size(); ++i) {
    float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
    float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
    float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
    float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

    if(res[i][res[i][4] + 6] > CONF ) {
      int type = res[i][4];
      string classname = classes[type];

      if (type==0) {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
      }
      else if (type==1) {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
      }
      else {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
      }
    }
  }
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO(vart::Runner* runner) {
  /* mean values for YOLO-v3 */
  float mean[3] = {0.0f, 0.0f, 0.0f};
  auto  inputTensors = cloneTensorBuffer(runner->get_input_tensors());
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;
  auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
  // input/output data define
  float *data = new float[shapes.inTensorList[0].size * inputTensors[0]->get_dim_size(0)];
  float *result0 = new float[shapes.outTensorList[0].size * outputTensors[shapes.output_mapping[0]]->get_dim_size(0)];
  float *result1 = new float[shapes.outTensorList[1].size * outputTensors[shapes.output_mapping[1]]->get_dim_size(0)];
  float *result2 = new float[shapes.outTensorList[2].size * outputTensors[shapes.output_mapping[2]]->get_dim_size(0)];
  float *result3 = new float[shapes.outTensorList[3].size * outputTensors[shapes.output_mapping[3]]->get_dim_size(0)];
  vector<float*> result;
  result.push_back(result0);
  result.push_back(result1);
  result.push_back(result2);
  result.push_back(result3);
  std::vector<std::unique_ptr<vart::TensorBuffer> > inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  while (true) {
    pair<int, Mat> pairIndexImage;

    mtxQueueInput.lock();
    if (queueInput.empty()) {
      mtxQueueInput.unlock();
      if (bReading)
      {
        continue;
      } else {
        break;
      }
    } else {
      /* get an input frame from input frames queue */
      pairIndexImage = queueInput.front();
      queueInput.pop();
      mtxQueueInput.unlock();
    }
    /* feed input frame into DPU Task with mean value */
    setInputImageForYOLO(runner, data, pairIndexImage.second, mean);
    // input/output tensorbuffer prepare
    inputs.push_back(
      std::make_unique<CpuFlatTensorBuffer>(data,inputTensors[0].get()));
    outputs.push_back( std::make_unique<CpuFlatTensorBuffer>(
      result0,outputTensors[shapes.output_mapping[0]].get()));
    outputs.push_back( std::make_unique<CpuFlatTensorBuffer>(
      result1,outputTensors[shapes.output_mapping[1]].get()));
    outputs.push_back( std::make_unique<CpuFlatTensorBuffer>(
      result2,outputTensors[shapes.output_mapping[2]].get()));
    outputs.push_back( std::make_unique<CpuFlatTensorBuffer>(
      result3,outputTensors[shapes.output_mapping[3]].get()));
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.resize(4);
    outputsPtr[shapes.output_mapping[0]] = outputs[0].get();
    outputsPtr[shapes.output_mapping[1]] = outputs[1].get();
    outputsPtr[shapes.output_mapping[2]] = outputs[2].get();
    outputsPtr[shapes.output_mapping[3]] = outputs[3].get();
    /* invoke the running of DPU for YOLO-v3 */
    auto job_id = runner->execute_async(inputsPtr,outputsPtr);
    runner->wait(job_id.first,-1);

    postProcess(runner, pairIndexImage.second, result, width, height);
    mtxQueueShow.lock();

    /* push the image into display frame queue */
    queueShow.push(pairIndexImage);
    mtxQueueShow.unlock();
    inputs.clear();
    outputs.clear();
    inputsPtr.clear();
    outputsPtr.clear();

  }
  delete[] data;
  delete[] result0;
  delete[] result1;
  delete[] result2;
  delete[] result3;
}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char** argv) {
  if (argc != 3) {
    cout << "Usage of ADAS detection: ./adas path(for json file) 1(usePPFlag)" << endl;
    return -1;
  }

  vector<string> images;
  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return -1;
  }
  runTotal = images.size();
  usePPFlag = stoi(argv[2])==1;
  /* Load xclbin for hardware pre-processor */
  if (usePPFlag){
    pp_kernel_init(handle,"/usr/lib/dpu.xclbin", "pp_pipeline_accel",0);
  }
  /* Create 1 DPU Tasks for YOLO-v3 network model */
  /* Spawn 3 threads:
     - 1 thread for reading video frame
     - 1 thread for running YOLO-v3 network model
     - 1 thread for displaying frame in monitor
     */
  auto graph = xir::Graph::deserialize(argv[1]);  
  auto subgraph = get_dpu_subgraph(graph.get());

  auto runnerptr = vart::Runner::create_runner(subgraph[0], "run");
  auto runner = runnerptr.get();
  // get in/out tenosrs
  auto  inputTensors = runner->get_input_tensors();
  auto  outputTensors = runner->get_output_tensors();
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  // init the shape info
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(
                 runner, &shapes, inputCnt,
                 {"layer81", "layer93", "layer105", "layer117"});

  array<thread, 3> threadsList = {
    thread(readFrame, images),
    thread(displayFrame, images),
    thread(runYOLO, runner)
  };

  for (int i = 0; i < 3; i++) {
    threadsList[i].join();
  }
  delete runner;

  return 0;
}

