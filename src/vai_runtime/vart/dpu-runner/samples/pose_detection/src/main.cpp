/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "pose.h"
#include "ssd.h"

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace detect;

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_displaying = true;

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display
// string pathBase;
string pose_model_path;
string ssd_model_path;
mutex mtx_create_runner;

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runGestureDetect(bool& is_running) {
  SSD ssd;
  GestureDetect gesture;
  {
    mtx_create_runner.lock();
    ssd.Init(ssd_model_path);
    gesture.Init(pose_model_path);
    mtx_create_runner.unlock();
  }
  // Run detection for images in read queue
  while (is_running) {
    // Get an image from read queue
    int index;
    Mat img;
    mtx_read_queue.lock();
    if (read_queue.empty()) {
      mtx_read_queue.unlock();
      if (is_reading) {
        continue;
      } else {
        is_running = false;
        break;
      }
    } else {
      index = read_queue.front().first;
      img = read_queue.front().second;
      read_queue.pop();
      mtx_read_queue.unlock();
    }

    // detect persons using ssd
    vector<tuple<int, float, cv::Rect_<float>>> results;
    ssd.Run(img, &results);

    // detect joint point of each person
    for (size_t i = 0; i < results.size(); ++i) {
      int xmin = get<2>(results[i]).x * img.cols;
      int ymin = get<2>(results[i]).y * img.rows;
      int xmax = xmin + (get<2>(results[i]).width) * img.cols;
      int ymax = ymin + (get<2>(results[i]).height) * img.rows;
      xmin = min(max(xmin, 0), img.cols);
      xmax = min(max(xmax, 0), img.cols);
      ymin = min(max(ymin, 0), img.rows);
      ymax = min(max(ymax, 0), img.rows);
      Rect roi = Rect(Point(xmin, ymin), Point(xmax, ymax));
      Mat sub_img = img(roi);
      gesture.Run(sub_img);
    }
    // Put image into display queue
    mtx_display_queue.lock();
    display_queue.push(make_pair(index, img));
    mtx_display_queue.unlock();
  }

  ssd.Finalize();
  gesture.Finalize();
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool& is_reading) {
  while (is_reading) {
    Mat img;
    if (read_queue.size() < 30) {
      if (!video.read(img)) {
        cout << "Finish reading the video." << endl;
        is_reading = false;
        break;
      }
      mtx_read_queue.lock();
      read_queue.push(make_pair(read_index++, img));
      mtx_read_queue.unlock();
    } else {
      usleep(20);
    }
  }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
  while (is_displaying) {
    mtx_display_queue.lock();
    if (display_queue.empty()) {
      if (is_running_1) {
        mtx_display_queue.unlock();
        usleep(20);
      } else {
        mtx_display_queue.unlock();
        is_displaying = false;
        break;
      }
    } else if (display_index == display_queue.top().first) {
      // Display image
      imshow("PoseDetection @Xilinx DPU", display_queue.top().second);
      display_index++;
      display_queue.pop();
      mtx_display_queue.unlock();
      if (waitKey(1) == 'q') {
        is_reading = false;
        is_running_1 = false;
        is_displaying = false;
        break;
      }
    } else {
      mtx_display_queue.unlock();
    }
  }
}

/**
 * @brief Entry for running pose detection neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char** argv) {
  // Check args
  if (argc != 4) {
    cout << "Usage of pose detection demo: " << argv[0] 
         << " <video_file> <pose_model_file> <ssd_model_file>"
         << endl;
    return -1;
  }

  // Initializations
  string file_name = argv[1];
  pose_model_path = argv[2];
  ssd_model_path = argv[3];
  cout << "Detect video: " << file_name << endl;
  video.open(file_name);
  if (!video.isOpened()) {
    cout << "Failed to open video: " << file_name;
    return -1;
  }
  // Run tasks
  array<thread, 4> threads = {thread(Read, ref(is_reading)),
                              thread(runGestureDetect, ref(is_running_1)),
                              thread(runGestureDetect, ref(is_running_1)),
                              thread(Display, ref(is_displaying))};

  for (int i = 0; i < 4; ++i) {
    threads[i].join();
  }

  // Detach from DPU driver and release resources
  video.release();

  return 0;
}
