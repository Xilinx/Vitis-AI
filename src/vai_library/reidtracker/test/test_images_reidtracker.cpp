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

#include <zconf.h>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <ratio>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

#include <vitis/ai/refinedet.hpp>
#include <vitis/ai/reid.hpp>
#include <vitis/ai/reidtracker.hpp>
#include "../src/common.hpp"

using namespace cv;
using namespace vitis::ai;
using namespace std;
using namespace vitis;
using namespace std::chrono;

long imgCount = 0;
int idxInputImage = 1;  // image index of input video
int idxShowImage = 1;   // next frame index to be display
bool bReading = true;   // flag of input
chrono::system_clock::time_point start_time;
int flag = 0;

typedef pair<int, Mat> imagePair;
class paircomp {
 public:
  bool operator()(const imagePair &n1, const imagePair &n2) const {
    return n1.first > n2.first;
  }
};

mutex mtxQueueInput;          // mutex of input queue
mutex mtxQueueShow;           // mutex of display queue
queue<imagePair> queueInput;  // input queue
priority_queue<imagePair, vector<imagePair>, paircomp>
    queueShow;  // display queue

/* Base parameter and image path.*/
string baseImagePath;
vector<string> images;
vector<Mat> read_imgs;
void reader() {
  for (size_t i = 0; i < images.size(); ++i) {
    string imageName = images[i];
    Mat img = imread(baseImagePath + "/img1/" + imageName);
    // resize(img, img, Size(480, 360));
    read_imgs.emplace_back(img);
  }
  cout << "read end " << endl;
  for (size_t i = 0; i < images.size(); ++i) {
    string imageName = images[i];
    if (queueInput.size() < 30) {
      Mat img;
      read_imgs[i].copyTo(img);
      mtxQueueInput.lock();
      queueInput.push(make_pair(idxInputImage++, img));
      mtxQueueInput.unlock();
      if (flag == 0) {
        start_time = chrono::system_clock::now();
        flag = 1;
      }
    }
    while (queueInput.size() >= 30) usleep(1000);
  }
  cout << "##############################video end, count: " << imgCount
       << endl;
  usleep(20000000);
}

string num2str(int i) {
  char ss[10];
  sprintf(ss, "%04d", i);
  return ss;
}

void displayImage() {
  Mat img;
  // VideoWriter writer;
  int imgcount = 1;
  // writer.open("results.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(960,
  // 540),
  //            true);
  while (true) {
    mtxQueueShow.lock();
    if (queueShow.empty()) {
      mtxQueueShow.unlock();
      usleep(10);
    } else if (idxShowImage == queueShow.top().first) {
      auto show_time = chrono::system_clock::now();
      stringstream buffer;
      Mat img = queueShow.top().second;
      auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
      buffer << fixed << setprecision(2)
             << (float)imgcount / (dura / 1000000.f);
      string a = buffer.str() + "FPS";
      cout << a << endl;
      resize(img, img, Size(960, 540));
      cv::putText(img, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},
                  1);
      cv::putText(img, to_string(queueShow.top().first), cv::Point(10, 1000), 1,
                  3, cv::Scalar{240, 240, 240}, 2);
      // resize(img, img, Size(720, 480));0
      // writer.write(img);
      // cv::imwrite(num2str(queueShow.top().first) + ".jpg", img);  // display
      // image moveWindow("reidvideo", 0, 0); cv::imshow("reidvideo", img);  //
      // display image waitKey(40);
      if (queueShow.top().first == imgCount) {
        // writer.release();
        exit(1);
      }

      idxShowImage++;
      imgcount++;
      cout << "idx show: " << idxShowImage << endl;
      if (idxShowImage == (imgCount + 1)) {
        exit(0);
        idxShowImage = 1;
      }
      queueShow.pop();
      mtxQueueShow.unlock();
    } else {
      mtxQueueShow.unlock();
    }
  }
}

void run() {
  auto tracker = vitis::ai::ReidTracker::create();
  auto reid = vitis::ai::Reid::create("personreid-res18_pt");
  auto det = vitis::ai::RefineDet::create("refinedet_pruned_0_96");
  std::vector<vitis::ai::ReidTracker::InputCharact> input_characts;
  string outfile = baseImagePath.substr(0, baseImagePath.size() - 1) + ".txt";
  ofstream of(outfile);
  while (1) {
    pair<int, Mat> pairIndexImage;
    mtxQueueInput.lock();
    if (queueInput.empty()) {
      mtxQueueInput.unlock();
      if (bReading)
        continue;
      else
        break;
    } else {
      // Get an image from input queue
      pairIndexImage = queueInput.front();
      queueInput.pop();
      mtxQueueInput.unlock();
    }
    Mat image = pairIndexImage.second;
    int frame_id = pairIndexImage.first;
    if (frame_id == 1) tracker = vitis::ai::ReidTracker::create();
    __TIC__(refinedet)
    auto results = det->run(image);
    __TOC__(refinedet)
    input_characts.clear();
    int lid = 0;
    for (auto box : results.bboxes) {
      Rect in_box = Rect(box.x * image.cols, box.y * image.rows,
                         box.width * image.cols, box.height * image.rows);
      Mat img = image(in_box);
      auto feat = reid->run(img).feat;
      input_characts.emplace_back(feat, in_box, box.score, -1, lid++);
    }
    // input_characts.resize(3);
    __TIC__(track)
    std::vector<vitis::ai::ReidTracker::OutputCharact> track_results =
        std::vector<vitis::ai::ReidTracker::OutputCharact>(
            tracker->track(frame_id, input_characts, true, true));
    __TOC__(track)
    __TIC__(deallater)
    cout << "frame_id: " << frame_id << endl;
    for (auto &r : track_results) {
      auto box = get<1>(r);
      float x = box.x;
      float y = box.y;
      float xmin = x;
      float ymin = y;
      float xmax = x + (box.width);
      float ymax = y + (box.height);
      xmin = std::min(std::max(xmin, 0.f), float(image.cols));
      xmax = std::min(std::max(xmax, 0.f), float(image.cols));
      ymin = std::min(std::max(ymin, 0.f), float(image.rows));
      ymax = std::min(std::max(ymax, 0.f), float(image.rows));
      uint64_t gid = get<0>(r);
      // float score = get<2>(r);
      of << frame_id << "," << gid << "," << xmin << "," << ymin << ","
         << xmax - xmin << "," << ymax - ymin << ",1,-1,-1,-1\n";
      cout << "RESULT: "
           << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax
           << "\t" << gid << "\n";
      cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                    cv::Scalar(255, 0, 0), 2, 1, 0);
      cv::putText(image, to_string(gid), cv::Point(xmin, ymin), 1, 2,
                  cv::Scalar{0, 0, 255}, 2);
    }
    __TOC__(deallater)
    if (frame_id == imgCount) {
      cout << "##########################finsh : " << frame_id << endl;
      of.close();
    }
    pairIndexImage.second = image;
    mtxQueueShow.lock();
    // Put the processed iamge to show queue
    queueShow.push(pairIndexImage);
    mtxQueueShow.unlock();
  }
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    cout << "Please set input image. " << endl;
    return 0;
  }
  baseImagePath = string(argv[1]);
  std::ifstream fs(baseImagePath + "/list.txt");
  std::string line;
  while (getline(fs, line)) {
    images.emplace_back(line);
  }
  imgCount = images.size();
  if (images.size() == 0) {
    cerr << "\nError: Not images exist in " << baseImagePath << endl;
    return -1;
  }
  fs.close();
  array<thread, 3> threads = {thread(run), thread(displayImage),
                              thread(reader)};
  for (int i = 0; i < 3; i++) {
    threads[i].join();
  }

  return 0;
}
