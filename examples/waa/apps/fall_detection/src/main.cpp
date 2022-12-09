/*
 * Copyright 2021 Xilinx Inc.
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
#include <queue>
#include <iterator>     // std::back_inserter
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <functional>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <aks/AksSysManagerExt.h>
#include <aks/AksNodeParams.h>
#include <aks/AksTensorBuffer.h>
#include <function_pool.hpp>
#include <opencv2/opencv.hpp>

using namespace std::chrono;

AKS::AIGraph *OFGraph;
AKS::AIGraph *OFInferenceGraph;

std::atomic<long> frameCount(0);
std::once_flag reset_timer_flag;
high_resolution_clock::time_point start_timer;
int stack_size = 10;
int batch_size = 4;
int sw_optical_flow = 0;
int frame_count=0;

struct content_pointer {
    bool is_video;
    cv::VideoCapture* videoCaptureObj;
    boost::filesystem::directory_iterator dirIt;
};

// OpenCV videocapture objects and corresponding index
static std::map<std::string, content_pointer> map_content;

// Utility function to convert Mat object to TensorBuffer
void mat2TensorBuffer(const cv::Mat& src,
                      AKS::AksTensorBuffer* tensorBuffer) {
    assert(src.depth() == CV_8U);
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    uint8_t* srcptr = src.data;
    uint8_t* bufferData = reinterpret_cast<uint8_t*>(tensorBuffer->data().first);
    std::memcpy(bufferData, srcptr, rows*cols*channels*sizeof(uint8_t));
}


void loadKernels(std::vector<std::string>& kernelPaths) {
    AKS::SysManagerExt *sysMan = AKS::SysManagerExt::getGlobal();
    for (auto & kernelPath : kernelPaths) {
        sysMan->loadKernels(kernelPath);
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


// Run inference on a blob of Optical Flow vectors
// Get data from last n OF flow vectors, pass them for inference
void runOFInference(
        std::vector<
            std::pair<std::string,
                      std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>>>>& ofQueue,
        std::mutex &ofQueueMutex, std::atomic<bool> &is_done) {
    auto sysMan = AKS::SysManagerExt::getGlobal();
    int stackSize = stack_size + batch_size - 1;
    std::vector<std::pair<std::string, std::unique_ptr<vart::TensorBuffer>>> stacks;
    stacks.reserve(stackSize);
    while ((!is_done) | (!ofQueue.empty())) {
        // Locking of ofQueueMutex is not required above (before while loop)
        // since there won't be a race condition as we are not pushing anything
        // to ofQueue after is_done is set to True
        std::unique_lock<std::mutex> lock(ofQueueMutex);
        while ((! ofQueue.empty()) & (stacks.size() < stackSize)) {
            stacks.push_back({
                ofQueue.front().first,
                std::move(ofQueue.front().second.get().front())});
            ofQueue.erase(ofQueue.begin());
        }
        lock.unlock();

     
        if (stacks.size() == stackSize) {
            std::vector<std::unique_ptr<vart::TensorBuffer>> enqueueVec;
            enqueueVec.reserve(stackSize);
            std::vector<std::string> enqueueTitles;
            enqueueTitles.reserve(batch_size);

            for (int i=0; i<stackSize; i++) {
                auto tempPtr = std::make_unique<AKS::AksTensorBuffer>(
                    *(static_cast<AKS::AksTensorBuffer*>(stacks.at(i).second.get())));
                enqueueVec.push_back(std::move(tempPtr));
                if (i >= stackSize - batch_size)
                    enqueueTitles.push_back(stacks.at(i).first);
            }
            // Remove the first 4 elements in stacks (9 stacks elems will remain)
            stacks.erase(stacks.begin(), stacks.begin()+batch_size);

            frameCount += batch_size;
            auto inferenceFut = sysMan->enqueueJob(
                OFInferenceGraph, enqueueTitles, std::move(enqueueVec));
            auto future = std::move(inferenceFut.get());
            // auto probsBuffer = static_cast<AKS::AksTensorBuffer*>(future.at(0).release());
            // float* probsData = reinterpret_cast<float*>(probsBuffer->data().first);
            // for (int i=0; i<batch_size; i++) {
            //     std::cout << enqueueTitles.at(i) << " | Probability: " << probsData[i] << std::endl;
            // }
        }
        else if (is_done)
            break;
    }
}


// Run Optical Flow on a pair of images <current frame, previous frame>
std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>> runOpticalFlow(
        std::string title,
        std::unique_ptr<AKS::AksTensorBuffer> prev_tb,
        std::unique_ptr<AKS::AksTensorBuffer> curr_tb) {
    auto sysMan = AKS::SysManagerExt::getGlobal();
    std::vector<std::unique_ptr<vart::TensorBuffer>> vec;
    vec.reserve(2);
    vec.push_back(std::move(curr_tb));
    vec.push_back(std::move(prev_tb));
    // Optical Flow: EnqueueJob to SystemManager with Optical flow graph
    std::call_once(reset_timer_flag,
        [&sysMan](){sysMan->resetTimer();
                    start_timer = high_resolution_clock::now();});
    auto fut = sysMan->enqueueJob(OFGraph, title, std::move(vec));
    return fut;
}


// Push each image in the directory to Optical Flow and Inference jobs
void readImgDir(const std::string& container_name) {
    boost::filesystem::directory_iterator it_dir = map_content[container_name].dirIt;
    std::vector<boost::filesystem::path> vec;
    copy(it_dir, boost::filesystem::directory_iterator(), std::back_inserter(vec));
    sort(vec.begin(), vec.end());
    std::vector<boost::filesystem::path>::const_iterator vecIterator = vec.begin();
    std::unique_ptr<AKS::AksTensorBuffer> curr_tb_for_later(nullptr);
    std::unique_ptr<AKS::AksTensorBuffer> prev_tb(nullptr);
    std::cout << "[INFO] Started processing stream: " << container_name << std::endl;
    std::vector<std::pair<
        std::string,
        std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>>>> ofQueue;
    std::mutex ofQueueMutex;
    std::atomic<bool> is_done = false;
    std::thread inferenceThread(runOFInference, std::ref(ofQueue),
                                std::ref(ofQueueMutex), std::ref(is_done));
    while (true) {
        if (vecIterator == vec.end()) {  // end of the files in directory
            break;
        }
        std::string title = (*(vecIterator++)).string();
        
        cv::Mat curr_img = cv::imread(title,0);
        cv::resize(curr_img, curr_img, cv::Size(224,224), 0, 0);
        auto curr_tb = std::make_unique<AKS::AksTensorBuffer>(
            xir::Tensor::create(
                "mainImageTensor",
                {1, curr_img.rows, curr_img.cols, curr_img.channels()},
                xir::create_data_type<unsigned char>()));
        mat2TensorBuffer(curr_img, curr_tb.get());
        if (! prev_tb) {
            prev_tb = std::make_unique<AKS::AksTensorBuffer>(*curr_tb.get());
            continue;
        }
        curr_tb_for_later = std::make_unique<AKS::AksTensorBuffer>(*curr_tb.get());
        auto fut = runOpticalFlow(title, std::move(prev_tb), std::move(curr_tb));
        prev_tb = std::move(curr_tb_for_later);
        {
            std::lock_guard<std::mutex> lock(ofQueueMutex);
            ofQueue.push_back({title, std::move(fut)});
        }
    }
    is_done = true;
    inferenceThread.join();
    std::cout << "[INFO] Done processing stream: " << container_name << std::endl;
}


// Push each frame in the video to Optical Flow job
void readVideoObj(const std::string& container_name) {
    cv::VideoCapture* const vCaptureObj = map_content[container_name].videoCaptureObj;
    int frame = 1;
    std::unique_ptr<AKS::AksTensorBuffer> curr_tb_for_later(nullptr);
    std::unique_ptr<AKS::AksTensorBuffer> prev_tb(nullptr);
    std::cout << "Started processing stream: " << container_name << std::endl;
    std::vector<std::pair<
        std::string,
        std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>>>> ofQueue;
    std::mutex ofQueueMutex;
    std::atomic<bool> is_done = false;
    std::thread inferenceThread(runOFInference, std::ref(ofQueue),
                                std::ref(ofQueueMutex), std::ref(is_done));
    while (1) {
        cv::Mat curr_frame;
        if (!vCaptureObj->read(curr_frame)) {  // end of frames in video
            vCaptureObj->release();
            break;
        }
        std::string title = container_name + "/" + std::to_string(frame++) + ".jpg";
        auto curr_tb = std::make_unique<AKS::AksTensorBuffer>(
            xir::Tensor::create(
                "mainVideoTensor",
                {1, curr_frame.rows, curr_frame.cols, curr_frame.channels()},
                xir::create_data_type<unsigned char>()));
        mat2TensorBuffer(curr_frame, curr_tb.get());
        if (! prev_tb) {
            prev_tb = std::make_unique<AKS::AksTensorBuffer>(*curr_tb.get());
            continue;
        }
        curr_tb_for_later = std::make_unique<AKS::AksTensorBuffer>(*curr_tb.get());
        auto fut = runOpticalFlow(title, std::move(prev_tb), std::move(curr_tb));
        {
            std::lock_guard<std::mutex> lock(ofQueueMutex);
            ofQueue.push_back({title, std::move(fut)});
        }
    }
    is_done = true;
    inferenceThread.join();
    std::cout << "Done processing stream: " << container_name << std::endl;
}


// read worker
void Read(std::string container_name) {
    if (map_content[container_name].is_video)
        return readVideoObj(container_name);
    else
        return readImgDir(container_name);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: .exe <directory> <num_threads> " << std::endl;
        std::cout << "Pass directory that contains videos and/or directory of images" << std::endl;
        std::cout << "Try ./run.sh -h to get correct usage. Exiting ..." << std::endl;
        exit(1);
    }
    const std::string vDirPath = argv[1];
    int num_threads = std::stoi(argv[2]);
    sw_optical_flow= std::stoi(argv[3]);

    auto sysMan = AKS::SysManagerExt::getGlobal();
    std::vector<std::string> kernelPaths;
    kernelPaths.push_back("kernel_zoo");
    // kernelPaths.push_back(std::string(std::getenv("AKS_ROOT"))+"/kernel_zoo");
    loadKernels(kernelPaths);
    std::string OFGraphJson;
    if(sw_optical_flow==0)
      OFGraphJson = "graph_zoo/graph_optical_flow.json";
    else  
      OFGraphJson = "graph_zoo/graph_optical_flow_opencv.json";
    
    std::string OFGraphName = "optical_flow";
    loadGraph(OFGraphJson, OFGraphName, &OFGraph);

    std::string OFInferGraphJson = "graph_zoo/graph_of_inference.json";
    std::string OFInferGraphName = "of_inference";
    loadGraph(OFInferGraphJson, OFInferGraphName, &OFInferenceGraph);

    std::vector<std::thread> mainThreads;
    std::string container_name;

    Function_pool func_pool(num_threads, Read);
    boost::filesystem::directory_iterator dit {vDirPath};
    std::vector<boost::filesystem::path> vec;
    copy(dit, boost::filesystem::directory_iterator(), std::back_inserter(vec));
    sort(vec.begin(), vec.end());
    for(auto vecIterator=vec.begin(); vecIterator != vec.end(); vecIterator++)
    {
        std::string dirElement = vecIterator->string();
        cv::VideoCapture *videoCaptureObj = new cv::VideoCapture();
        videoCaptureObj->open(dirElement);
        boost::filesystem::directory_iterator subDirIterator {dirElement};
        content_pointer cp {false, nullptr, subDirIterator};
        std::string container_name = vecIterator->filename().string();
        if (!videoCaptureObj->isOpened()) {
            videoCaptureObj->release();
        }
        else {
            cp.is_video = true;
            cp.videoCaptureObj = videoCaptureObj;
        }
        map_content[container_name] = cp;
        func_pool.push(container_name);
    }
    func_pool.done();
    func_pool.wait_on_thread_pool();
    auto stop_timer = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop_timer - start_timer);
    sysMan->report(OFInferenceGraph);
    std::cout << "Total frames inferred: " << frameCount << std::endl;
    std::cout << "Total timetaken: " << (float)duration.count()/1000000 << " seconds.." << std::endl;
    std::cout << "Throughput (fps): " << frameCount*1000000/(float)duration.count() << std::endl;
    AKS::SysManagerExt::deleteGlobal();
    return 0;
}

// info proc status
