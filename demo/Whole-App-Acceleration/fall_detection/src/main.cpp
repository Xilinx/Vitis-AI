/*
 * Copyright 2020 Xilinx Inc.
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

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iterator>     // std::back_inserter
#include <algorithm>
#include <condition_variable>
#include <chrono>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <aks/AksSysManagerExt.h>
#include <aks/AksNodeParams.h>

using namespace std::chrono;

AKS::AIGraph *OFGraph;
AKS::AIGraph *OFInferenceGraph;

std::atomic<long> frameCount(0);
std::once_flag reset_timer_flag;
high_resolution_clock::time_point start_timer;

struct content_pointer {
    bool is_video;
    cv::VideoCapture* videoCaptureObj;
    boost::filesystem::directory_iterator dirIt;
};

static std::map<std::string, bool> status_flag;  // status flag by stream
static std::map<std::string, content_pointer> map_content;  // OpenCV videocapture objects and corresponding index
std::mutex mtx_sflag;  // mutex of status_flag map

// Utility function to convert Mat object to DataDescriptor
AKS::DataDescriptor mat2DD(const cv::Mat& src) {
    assert(src.depth() == CV_8U);
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    AKS::DataDescriptor dst({1, channels, rows, cols}, AKS::DataType::UINT8);
    uint8_t* dstptr = dst.data<uint8_t>();
    uint8_t* srcptr = src.data;
    std::memcpy(dstptr, srcptr, channels*rows*cols*sizeof(uint8_t));
    return dst;
}


void loadKernels(std::vector<std::string>& kernelPaths) {
    AKS::SysManagerExt *sysMan = AKS::SysManagerExt::getGlobal();
    for (auto & kernelPath : kernelPaths) {
        sysMan -> loadKernels(kernelPath);
    }
}


// load AIGraph given graph json and graph name
void loadGraph(std::string& graphJson, std::string& graphName, AKS::AIGraph **graph)
{
    AKS::SysManagerExt *sysMan = AKS::SysManagerExt::getGlobal();
    sysMan->loadGraphs(graphJson);
    *graph = sysMan->getGraph(graphName);
    if(!graph) {
        std::cout << "[ERROR] Couldn't find requested graph" << std::endl;
        AKS::SysManagerExt::deleteGlobal();
    }
}


AKS::DataDescriptor createBlob(std::vector<AKS::DataDescriptor>& vDD) {
  uint8_t numDD = vDD.size();
  assert(numDD > 0);

  auto dd = vDD[0];
  auto shape = dd.getShape();
  int channels = shape[1];
  int rows = shape[2];
  int cols = shape[3];

  AKS::DataDescriptor outDD({1, channels*numDD, rows, cols}, AKS::DataType::FLOAT32);
  float* outData = outDD.data<float>();
  // merge at Channels
  for (int i=0; i<numDD*channels; i++) {
    int8_t* data = vDD[i/channels].data<int8_t>();
    for (int j=0; j<rows; j++) {
      for (int k=0; k<cols; k++) {
          outData[i*rows*cols + j*cols + k] = data[(i%channels)*rows*cols + j*cols + k];
      }
    }
  }
  return outDD;
}


// Run inference on a blob of Optical Flow vectors
// Get data from last n OF flow vectors, pass them for inference
void runOFInference(std::vector<AKS::DataDescriptor>& stack,
        std::future<std::vector<AKS::DataDescriptor>> fut, std::string title,
        std::mutex& stack_mutex, int counter, int& whose_turn,
        std::condition_variable &cond_var) {
    int stack_size=10;
    int stride=1;
    auto sysMan = AKS::SysManagerExt::getGlobal();
    AKS::DataDescriptor outDD = fut.get()[0];
    std::unique_lock<std::mutex> ulock(stack_mutex);
    cond_var.wait(ulock, [counter, &whose_turn] {return counter == whose_turn;});
    whose_turn++;
    stack.push_back(outDD);
    if (stack.size() >= stack_size) {
        std::vector<AKS::DataDescriptor> st(stack.begin(), stack.begin()+stack_size);
        stack.erase(stack.begin(), stack.begin()+stride);
        ulock.unlock();
        cond_var.notify_all();
        frameCount++;
        auto inferenceFut = sysMan->enqueueJob(OFInferenceGraph, title, std::move(st));
        auto outDD = inferenceFut.get()[0];
        float* outData = outDD.data<float>();
        std::cout << "Frame: " << title << " | Class: " << outData[0] << std::endl;
    }
    else {
        ulock.unlock();
        cond_var.notify_all();
    }
}


// Run Optical Flow on a pair of images <current frame, previous frame>
std::future<std::vector<AKS::DataDescriptor>> runOpticalFlow(
        std::string title, AKS::DataDescriptor& prev_dd,
        AKS::DataDescriptor& curr_dd) {
    auto sysMan = AKS::SysManagerExt::getGlobal();
    std::vector<AKS::DataDescriptor> v { curr_dd, prev_dd };
    // Optical Flow: EnqueueJob to SystemManager with Optical flow graph
    std::call_once(reset_timer_flag,
        [&sysMan](){sysMan->resetTimer();
                    start_timer = high_resolution_clock::now();});
    auto fut = sysMan->enqueueJob(OFGraph, title, std::move(v));
    return fut;
}


// Push each image in the directory to Optical Flow and Inference jobs
void readImgDir(std::string container_name) {
    boost::filesystem::directory_iterator it_dir = map_content[container_name].dirIt;
    AKS::DataDescriptor prev_dd;
    std::vector<AKS::DataDescriptor> stack;
    std::mutex stack_mutex;
    std::condition_variable cond_var;
    int counter = 0;
    int whose_turn = 0;

    std::vector<std::thread> threads;

    std::vector<boost::filesystem::path> vec;
    copy(it_dir, boost::filesystem::directory_iterator(), std::back_inserter(vec));
    sort(vec.begin(), vec.end());

    std::vector<boost::filesystem::path>::const_iterator vecIterator = vec.begin();
    while (1) {
        std::unique_lock<std::mutex> locker(mtx_sflag);
        std::map<std::string, bool>::iterator it = status_flag.find(container_name);
        if (it != status_flag.end()) {
            if (it->second == true) {            // would be false if user chose to abort this stream
                if (vecIterator == vec.end()) {  // end of the files in directory
                    status_flag.erase(it->first);
                    break;
                } else {
                    locker.unlock();
                    std::string title = (*(vecIterator++)).string();
                    cv::Mat curr_image = cv::imread(title);
                    AKS::DataDescriptor curr_dd = mat2DD(curr_image);
                    if (prev_dd.getNumberOfElements() == 0) {
                        prev_dd = curr_dd;
                        continue;
                    }
                    auto fut = runOpticalFlow(title, prev_dd, curr_dd);
                    auto th = std::thread(
                        runOFInference, std::ref(stack), std::move(fut), title,
                        std::ref(stack_mutex), counter++, std::ref(whose_turn),
                        std::ref(cond_var));
                    threads.push_back(std::move(th));
                    prev_dd = curr_dd;
                }
            } else {  // User chose to abort this stream
                status_flag.erase(it->first);
                std::cout << "=== [READ] Force exit reading directory: " << container_name << std::endl; // User requested input reading thread exit
                break;
            }
        }
    }
    for (auto & th: threads)
        th.join();
}


// Push each frame in the video to Optical Flow job
void readVideoObj(std::string container_name) {
    cv::VideoCapture* const vCaptureObj = map_content[container_name].videoCaptureObj;
    int frame = 1;
    AKS::DataDescriptor prev_dd;
    std::vector<AKS::DataDescriptor> stack;
    std::mutex stack_mutex;
    std::condition_variable cond_var;
    int counter = 0;
    int whose_turn = 0;
    std::vector<std::thread> threads;

    while (1) {
        std::unique_lock<std::mutex> locker(mtx_sflag);
        std::map<std::string, bool>::iterator it = status_flag.find(container_name);
        if (it != status_flag.end()) {
            if (it->second == true) {  // would be false if user chose to abort this stream
                cv::Mat curr_frame;
                if (!vCaptureObj->read(curr_frame)) {  // end of frames in video
                    status_flag.erase(it->first);
                    vCaptureObj->release();
                    break;
                } else {
                    locker.unlock();
                    std::string title = container_name + "/" + std::to_string(frame++) + ".jpg";
                    AKS::DataDescriptor curr_dd = mat2DD(curr_frame);
                    if (prev_dd.getNumberOfElements() == 0) {
                        prev_dd = curr_dd;
                        continue;
                    }
                    auto fut = runOpticalFlow(title, prev_dd, curr_dd);
                    auto th = std::thread(
                        runOFInference, std::ref(stack), std::move(fut), title,
                        std::ref(stack_mutex), counter++, std::ref(whose_turn),
                        std::ref(cond_var));
                    threads.push_back(std::move(th));
                    prev_dd = curr_dd;
                }
            } else {  // User chose to abort this stream
                status_flag.erase(it->first);
                vCaptureObj->release();
                std::cout << "=== [READ] Force exit reading video: " << container_name << std::endl; // User requested input reading thread exit
                break;
            }
        }
    }
    for (auto & th: threads)
        th.join();
}

// read worker
void Read(std::string container_name) {
    if (map_content[container_name].is_video)
        return readVideoObj(container_name);
    else
        return readImgDir(container_name);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: .exe <directory>" << std::endl;
        std::cout << "Pass directory that contains videos and/or directory of images" << std::endl;
    }
    const std::string vDirPath = argv[1];

    auto sysMan = AKS::SysManagerExt::getGlobal();
    std::vector<std::string> kernelPaths;
    kernelPaths.push_back("kernel_zoo");
    kernelPaths.push_back(std::string(std::getenv("AKS_ROOT"))+"/kernel_zoo");
    loadKernels(kernelPaths);

    std::string OFGraphJson = "graph_zoo/graph_optical_flow.json";
    std::string OFGraphName = "optical_flow";
    loadGraph(OFGraphJson, OFGraphName, &OFGraph);

    std::string OFInferGraphJson = "graph_zoo/graph_of_inference.json";
    std::string OFInferGraphName = "of_inference";
    loadGraph(OFInferGraphJson, OFInferGraphName, &OFInferenceGraph);

    std::vector<std::thread> mainThreads;
    std::string container_name;
    for(boost::filesystem::directory_iterator dit {vDirPath};
            dit != boost::filesystem::directory_iterator{}; dit++)
    {
        std::string dirElement = dit->path().string();
        cv::VideoCapture *videoCaptureObj = new cv::VideoCapture();
        videoCaptureObj->open(dirElement);
        content_pointer cp;
        std::string container_name = dit->path().filename().string();
        if (!videoCaptureObj->isOpened()) {
            cp.is_video = false;
            boost::filesystem::directory_iterator subDirIterator {dirElement};
            cp.dirIt = subDirIterator;
        }
        else {
            cp.is_video = true;
            cp.videoCaptureObj = videoCaptureObj;
        }
        map_content[container_name] = cp;
        status_flag[container_name] = true;
        mainThreads.push_back(std::thread(Read, container_name));
    }
    for (auto & thread : mainThreads) {
        thread.join();
    }
    auto stop_timer = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop_timer - start_timer);
    std::cout << "Total timetaken: " << (float)duration.count()/1000000 << " seconds.." << std::endl;
    sysMan->report(OFInferenceGraph);
    std::cout << "Throughput (fps): " << frameCount*1000000/(float)duration.count() << std::endl;
    AKS::SysManagerExt::deleteGlobal();
    return 0;
}
