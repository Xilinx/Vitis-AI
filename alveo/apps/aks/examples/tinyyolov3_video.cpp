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
#include <atomic>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ext/AksSysManagerExt.h"
#include "ext/AksNodeParams.h"

using namespace std;
using namespace cv;
using namespace AKS;

struct futures_queue_element {
    cv::Mat frame;
    std::future<std::vector<AKS::DataDescriptor>> futureObj;
};

std::queue<futures_queue_element> futures_queue;
std::mutex mtx_futures_queue;                        // mutex of futures_queue
std::atomic<bool> done_reading_video(false);         // Done pushing frames to read_queue

VideoWriter video;
bool doResize;
// if doResize is set, output frame is resized to (outHeight, outWidth)
int outHeight;
int outWidth;


// Utility function to convert Mat object to DataDescriptor
AKS::DataDescriptor mat2DD(const cv::Mat& src) {
    assert(src.depth() == CV_8U);
    AKS::DataDescriptor dst({1, src.channels(), src.rows, src.cols}, AKS::DataType::UINT8);
    uint8_t* dstptr = dst.data<uint8_t>();
    uint8_t* srcptr = src.data;
    std::memcpy(dstptr, srcptr, src.rows * src.cols * src.channels());
    return dst;
}

bool getFileContent(std::string fileName, std::vector<std::string> &vecOfLabels)
{
    std::ifstream in(fileName.c_str());
    if(!in)
    {
        std::cerr << "[ERROR] FileNotFound: Cannot find the file: " << fileName << std::endl;
        return false;
    }
    std::string str;
    while (std::getline(in, str))
        if(str.size() > 0)
            vecOfLabels.push_back(str);
    in.close();
    return true;
}


void writeBboxVideo(cv::Mat &frame, AKS::DataDescriptor &dd,
                    std::vector<std::string> &vecOfLabels) {
    int nboxes = dd.getShape()[0];
    int coords = 6;
    float* outData = (float*) dd.data();

    float resizeHeightFactor;
    float resizeWidthFactor;
    cv::Mat image;

    if (!doResize) {
        image = frame;
        resizeHeightFactor = 1;
        resizeWidthFactor = 1;
    }
    else {
        resizeHeightFactor = ((float)frame.rows)/outHeight;
        resizeWidthFactor = ((float)frame.cols)/outWidth;

        image = cv::Mat(outHeight, outWidth, CV_8UC3);
        cv::resize(frame, image, cv::Size(outWidth, outHeight));
    }

    for (int i=0; i<nboxes; i++) {
        float score = outData[i*coords+5];
        int left = (int) outData[i*coords]/resizeWidthFactor;
        int right = (int) outData[i*coords+2]/resizeWidthFactor;

        int bottom = (int) outData[i*coords+1]/resizeHeightFactor;
        int top = (int) outData[i*coords+3]/resizeHeightFactor;

        int classID = (int) outData[i*coords+4];
        cv::rectangle(image, cv::Point(left, top),
                      cv::Point(right, bottom), cv::Scalar(255,0,0), 2);
        std::string label = vecOfLabels[classID];
        cv::putText(image, label, cv::Point(left, top),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,250), 2);
    }
    video.write(image);
}


void processAndWrite(AKS::AIGraph *graph, std::vector<std::string>& vecOfLabels) {
    while (1) {
        std::unique_lock<std::mutex> locker(mtx_futures_queue);
        if (done_reading_video.load() && futures_queue.empty())
            return;
        if (!futures_queue.empty())
        {
            auto frame = futures_queue.front().frame;
            auto outDD = futures_queue.front().futureObj.get();
            futures_queue.pop();
            locker.unlock();
            writeBboxVideo(frame, outDD[0], vecOfLabels);
        }
    }
}


void loadKernels(std::vector<std::string> kernelPaths) {
    AKS::SysManagerExt *sysMan = AKS::SysManagerExt::getGlobal();
    for (auto & kernelPath : kernelPaths) {
        sysMan->loadKernels(kernelPath);
    }
}


void loadGraph(std::string& graphJson, std::string& graphName, AKS::AIGraph ** graph) {
    AKS::SysManagerExt *sysMan = AKS::SysManagerExt::getGlobal();
    sysMan->loadGraphs(graphJson);
    *graph = sysMan->getGraph(graphName);
    if(!graph) {
        std::cout << "[ERROR] Couldn't find requested graph" << std::endl;
        AKS::SysManagerExt::deleteGlobal();
    }
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: .exe <video file>" << std::endl;
        std::cout << "Pass video file" << std::endl;
        return 0;
    }
    const std::string videoFile = argv[1];
    const std::string output_video = "./detection_output.avi";
    std::string labelsFile = std::string(getenv("VAI_ALVEO_ROOT")) + \
        "/apps/yolo/coco.names";
    std::vector<std::string> vecOfLabels;
    if (! getFileContent(labelsFile, vecOfLabels)) {
        return 0;
    }
    // doResize = false;
    // outHeight = 1080;
    // outWidth = 1920;

    doResize = true;  // This is to compress output frame resolution as (ultra-)HD
                      // videos are compressed by video players to fit to the
                      // screen resoultion which leads to observing incomplete
                      // bounding boxes
    outHeight = 540;  // Chosen arbitrarily, can be updated
    outWidth = 960;   // Chosen arbitrarily, can be updated

    // Get AKS System Manager instance
    std::string graphJson = "graph_zoo/graph_tinyyolov3_video.json";
    std::string graphName = "tinyyolov3";
    AKS::AIGraph *graph;

    auto sysMan = AKS::SysManagerExt::getGlobal();
    loadKernels({"kernel_zoo"});
    loadGraph(graphJson, graphName, &graph);

    VideoCapture *videoCaptureObj = new VideoCapture();
    videoCaptureObj->open(videoFile);

    if(!videoCaptureObj->isOpened()){
        cout << "Error opening video stream or file" << endl;
        return 0;
    }

    double fps = videoCaptureObj->get(CAP_PROP_FPS);
    video = VideoWriter(output_video, CV_FOURCC('M','J','P','G'), fps, Size(outWidth, outHeight));

    auto processThread = std::thread(processAndWrite, graph, std::ref(vecOfLabels));
    std::cout << "[INFO] Processing video: " << videoFile << std::endl;
    Mat frame;
    while (videoCaptureObj->read(frame)) {
        std::vector<AKS::DataDescriptor> v { mat2DD(frame) };
        mtx_futures_queue.lock();
        futures_queue.push({std::move(frame), sysMan->enqueueJob(graph, "", std::move(v), nullptr)});
        mtx_futures_queue.unlock();
    }
    done_reading_video = true;
    videoCaptureObj->release();
    processThread.join();

    std::cout << "[INFO] Video Saved at: " << output_video << std::endl;
    AKS::SysManagerExt::deleteGlobal();
    return 0;
}
