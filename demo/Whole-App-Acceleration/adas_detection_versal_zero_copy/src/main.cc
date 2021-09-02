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
#include <dirent.h>
#include <sys/stat.h>
#include <zconf.h>

#include <algorithm>
#include <atomic>
#include <queue>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <zconf.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "utils.h"

#include "vart/runner_ext.hpp"
#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include <opencv2/opencv.hpp>

#include "pp_wrapper.h"

#define EN_BRKP 0

using namespace std;
using namespace cv;
using namespace std::chrono;

#define NMS_THRESHOLD 0.3f
bool usePPFlag = true;
int idxInputImage = 0; // frame index of input video
int idxShowImage = 0;  // next frame index to be displayed
bool bReading = true;  // flag of reding input frame
bool bExiting = false;
chrono::system_clock::time_point start_time;

long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0;

typedef pair<int, Mat> imagePair;
class paircomp
{
public:
  bool operator()(const imagePair &n1, const imagePair &n2) const
  {
    if (n1.first == n2.first)
    {
      return (n1.first > n2.first);
    }

    return n1.first > n2.first;
  }
};

static std::unique_ptr<vart::TensorBuffer> allocate_tensor_buffer(
    xclDeviceHandle h, xclBufferHandle bo, size_t offset,
    const xir::Tensor *tensor);

static std::unique_ptr<vart::TensorBuffer> allocate_tensor_buffer(
    xclDeviceHandle h, xclBufferHandle bo, size_t offset,
    const xir::Tensor *tensor)
{
  return vart::assistant::XrtBoTensorBuffer::create({h, bo}, tensor);
}

const string baseImagePath = "./data/";
/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images)
{
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode))
  {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr)
  {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr)
  {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN)
    {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png"))
      {
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
queue<int8_t *> queueInput_2;
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
PPHandle *pp_handle;

void setInputImageForYOLO(int8_t *data, const Mat &frame, float input_scale)
{
  /* mean values for YOLO-v3 */
  float mean[3] = {0.0f, 0.0f, 0.0f};
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;
  int size = shapes.inTensorList[0].size;
  int channel = 3;

  if (usePPFlag)
  {
    /* pre-processing for input frame using HW accelerated kernel */
    preprocess(pp_handle, (unsigned char *)frame.data, frame.rows, frame.cols, height, width);
  }
  else
  {
    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for (int b = 0; b < height; ++b)
    {
      for (int c = 0; c < width; ++c)
      {
        for (int a = 0; a < channel; ++a)
        {
          bb[b * width * channel + c * channel + a] =
              img_yolo.data[a * height * width + b * width + c];
        }
      }
    }

    for (int i = 0; i < size; ++i)
    {
      data[i] = int(bb.data()[i] * input_scale);
      if (data[i] < 0)
        data[i] = 127;
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
void readFrame(vector<string> images, vart::Runner *runner, float input_scale)
{
  auto inputTensors = cloneTensorBuffer8bit(runner->get_input_tensors());
  auto data = queueInput_2.front();
  start_time = chrono::system_clock::now();

  while (true)
  {
    if (idxInputImage < images.size())
    {
      Mat img;
      auto t1 = std::chrono::system_clock::now();
      img = imread(baseImagePath + images[idxInputImage]);
      auto t2 = std::chrono::system_clock::now();
      auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
      imread_time += value_t1.count();
      mtxQueueInput.lock();
      queueInput.push(make_pair(idxInputImage++, img));
      // std::cout << std::hex << "data ptr : " << &data[0] << "\n";
      /* feed input frame into DPU Task with mean value */
#if EN_BRKP
      auto pre_t1 = chrono::system_clock::now();
#endif
      setInputImageForYOLO(data, img, input_scale);
#if EN_BRKP
      auto pre_t2 = std::chrono::system_clock::now();
      auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2 - pre_t1);
      pre_time += prevalue_t1.count();
#endif
      auto pre_etime = chrono::system_clock::now();
      mtxQueueInput.unlock();
    }
  }

  bExiting = true;
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame(vector<string> images)
{
  Mat frame;
  while (true)
  {
    //std::cout << "*******display" << std::endl;
    mtxQueueShow.lock();

    if (queueShow.empty())
    {
      mtxQueueShow.unlock();
    }
    else if (idxShowImage == queueShow.top().first)
    {
      frame = queueShow.top().second;
      cv::imwrite("./output/" + images[idxShowImage], frame);
      idxShowImage++;
      queueShow.pop();
      mtxQueueShow.unlock();
      if (idxShowImage == runTotal)
      {
        bReading = false;

        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        stringstream buffer_str;
        buffer_str << fixed << setprecision(1)
                   << (float)runTotal / (dura / 1000000.f);
        string a = buffer_str.str() + " FPS";
        //std::cout << "Performance:" << a << std::endl;
        std::cout << "Performance: " << 1000000.0 / (float)((dura - imread_time) / runTotal) << "fps\n";

#if EN_BRKP
        std::cout << "imread latency: " << (float)(imread_time / runTotal) / 1000 << "ms\n";
        std::cout << "pre latency: " << (float)(pre_time / runTotal) / 1000 << "ms\n";
        std::cout << "exec latency: " << (float)(exec_time / runTotal) / 1000 << "ms\n";
        std::cout << "post latency: " << (float)(post_time / runTotal) / 1000 << "ms\n";
#endif
        exit(0);
      }
    }
    else
    {
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
void postProcess(vart::Runner *runner, Mat &frame, vector<int8_t *> results,
            vector<float> output_scale, int sWidth, int sHeight)
{
  const string classes[3] = {"car", "person", "cycle"};

  /* four output nodes of YOLO-v3 */
  // const string outputs_node[4] = {"layer81_conv", "layer93_conv",
  //                                    "layer105_conv", "layer117_conv"};
  vector<vector<float>> boxes;
  // auto  outputTensors = runner->get_output_tensors();
  for (int ii = 0; ii < 4; ii++)
  {
    int width = shapes.outTensorList[ii].width;
    int height = shapes.outTensorList[ii].height;
    int channel = shapes.outTensorList[ii].channel;
    int sizeOut = channel * width * height;
    vector<float> result(sizeOut);
    boxes.reserve(sizeOut);

    /* Store every output node results */
    get_output(results[ii], output_scale[ii], sizeOut, channel, height, width, result);

    /* Store the object detection frames as coordinate information  */
    detect(boxes, result, channel, height, width, ii, sHeight, sWidth);
  }

  /* Restore the correct coordinate frame of the original image */
  correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth,
                       sHeight);

  /* Apply the computation for NMS */
  vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

  float h = frame.rows;
  float w = frame.cols;

  for (size_t i = 0; i < res.size(); ++i)
  {
    float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
    float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
    float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
    float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

    //std::cout << "x,y,xmax,ymax: " << xmin << ", " << ymin << xmax << ", " << ymax << "\n";

    if (res[i][res[i][4] + 6] > CONF)
    {
      int type = res[i][4];
      string classname = classes[type];

      if (type == 0)
      {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
                  Scalar(0, 0, 255), 1, 1, 0);
      }
      else if (type == 1)
      {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
                  Scalar(255, 0, 0), 1, 1, 0);
      }
      else
      {
        rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
                  Scalar(0, 255, 255), 1, 1, 0);
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
void runYOLO(vart::RunnerExt *runner, vector<float> output_scale)
{
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;

  auto out_ptr1 = output_tensor_buffers[0]->data({0, 0, 0, 0});
  signed char *result0 = (signed char *)out_ptr1.first;
  auto out_ptr2 = output_tensor_buffers[1]->data({0, 0, 0, 0});
  signed char *result1 = (signed char *)out_ptr2.first;
  auto out_ptr3 = output_tensor_buffers[2]->data({0, 0, 0, 0});
  signed char *result2 = (signed char *)out_ptr3.first;
  auto out_ptr4 = output_tensor_buffers[3]->data({0, 0, 0, 0});
  signed char *result3 = (signed char *)out_ptr4.first;

  vector<int8_t *> result;
  result.push_back(result0);
  result.push_back(result1);
  result.push_back(result2);
  result.push_back(result3);

  while (true)
  {
    pair<int, Mat> pairIndexImage;
    mtxQueueInput.lock();
    if (queueInput.empty())
    {
      mtxQueueInput.unlock();
      if (bExiting)
        break;
      if (bReading)
        continue;
      else
        break;
    }
    else
    {
      /* get an input frame from input frames queue */
      pairIndexImage = queueInput.front();
      queueInput.pop();      
      mtxQueueInput.unlock();
    }

#if EN_BRKP
    auto exec_t1 = std::chrono::system_clock::now();
#endif

    if (!usePPFlag)   {
      for (auto &input : input_tensor_buffers)   {
        input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                     input->get_tensor()->get_shape()[0]);
      }
    }
    /*run*/
    auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
    runner->wait(job_id.first, -1);

    for (auto output : output_tensor_buffers)   {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

#if EN_BRKP
    auto exec_t2 = std::chrono::system_clock::now();
    auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
    exec_time += execvalue_t1.count();
#endif
    postProcess(runner, pairIndexImage.second, result, output_scale, width, height);
#if EN_BRKP
    auto post_t1 = std::chrono::system_clock::now();
    auto postvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(post_t1 - exec_t2);
    post_time += postvalue_t1.count();
#endif
    mtxQueueShow.lock();

    /* push the image into display frame queue */
    queueShow.push(pairIndexImage);
    mtxQueueShow.unlock();
  }
}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char **argv)
{
  if (argc != 3)
  {
    cout << "Usage of ADAS detection: ./adas [model] [1(usePPFlag)]" << endl;
    return -1;
  }

  vector<string> images;
  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0)
  {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return -1;
  }
  runTotal = images.size();
  usePPFlag = stoi(argv[2]) == 1;

  /* Create 1 DPU Tasks for YOLO-v3 network model */
  /* Spawn 3 threads:
  - 1 thread for reading video frame
  - 1 running YOLO-v3 network model
  - 1 thread for displaying frame in monitor
  */
  // auto runners = vart::Runner::create_dpu_runner(argv[1]);
  auto attrs = xir::Attrs::create();
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "yolov3 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  /*create runner*/
  std::unique_ptr<vart::RunnerExt> runner =
      vart::RunnerExt::create_runner(subgraph[0], attrs.get());

  // get in/out tenosrs
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  // init the shape info
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt,
                 {"layer81", "layer93", "layer105", "layer117"});

  auto input_tensor_buffers = runner->get_inputs();
  float mean[3] = {0.0f, 0.0f, 0.0f};

  uint64_t dpu_input_phy_addr = 0u;
	uint64_t dpu_input_size = 0u;
	std::tie(dpu_input_phy_addr, dpu_input_size) = input_tensor_buffers[0]->data_phy({0,0,0,0});

  auto dpu_runner_ext = dynamic_cast<vart::RunnerExt *>(runner.get());
  auto input_scale = vart::get_input_scale(dpu_runner_ext->get_input_tensors());  
  auto output_scale = vart::get_output_scale(dpu_runner_ext->get_output_tensors());

  uint64_t data_in = 0u;
  size_t size_in = 0u;
  std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector<int>{0, 0, 0, 0});
  CHECK_NE(size_in, 0u);
  signed char *data = (signed char *)data_in;

  queueInput_2.push(data);

  if (usePPFlag)
    pp_kernel_init(pp_handle, "/usr/lib/dpu.xclbin", dpu_input_phy_addr, mean, input_scale[0]);

  array<thread, 3> threadsList = {
      thread(readFrame, images, runner.get(), input_scale[0]),
      thread(displayFrame, images),
      thread(runYOLO, runner.get(), output_scale)};

  for (int i = 0; i < 3; i++)
    threadsList[i].join();

  if (usePPFlag)
    pp_dealloc(pp_handle);

  return 0;
}
