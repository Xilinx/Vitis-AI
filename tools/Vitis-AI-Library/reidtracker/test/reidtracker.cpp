/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/reidtracker.hpp>
#include <vitis/ai/refinedet.hpp>
using namespace std;
using namespace cv;
using ReidTrackerResult = std::vector<vitis::ai::ReidTracker::OutputCharact>;
namespace vitis {
namespace ai {

// Implement a class that cascades pedestrian detect and pose detect
struct RefineDetReidTracker {
  static std::unique_ptr<RefineDetReidTracker> create();
  RefineDetReidTracker();
  ReidTrackerResult run(const cv::Mat &input_image);
  int getInputWidth();
  int getInputHeight();

private:
  std::unique_ptr<vitis::ai::RefineDet> refinedet_;
  std::shared_ptr<vitis::ai::ReidTracker> reid_tracker_;
};
// A factory function to get a instance of derived classes of class
std::unique_ptr<RefineDetReidTracker> RefineDetReidTracker::create() {
  return std::unique_ptr<RefineDetReidTracker>(new RefineDetReidTracker());
}
int RefineDetReidTracker::getInputWidth() { return refinedet_->getInputWidth(); }
int RefineDetReidTracker::getInputHeight() { return refinedet_->getInputHeight(); }

RefineDetReidTracker::RefineDetReidTracker()
    : refinedet_{vitis::ai::RefineDet::create("refinedet_pruned_0_96")},
      reid_tracker_{vitis::ai::ReidTracker::create()}{}

ReidTrackerResult RefineDetReidTracker::run(const cv::Mat &input_image) {
  cv::Mat image;
  auto size = cv::Size(refinedet_->getInputWidth(), refinedet_->getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // run refinedet
  auto results = refinedet_->run(image);
  std::vector<vitis::ai::ReidTracker::InputCharact> input_characts;
  int lid = 0;
  for(auto box : results.bboxes){
    Rect2f in_box = Rect2f(box.x, box.y, box.width, box.height);
    input_characts.emplace_back(in_box, box.score, -1, lid++);
      //cout<<"score: "<<box[6]<<" "<<in_box.x<<" "<<in_box.y<<" "<<in_box.width<<" "<<in_box.height<<" "<<lid<<endl;
    }
    std::vector<vitis::ai::ReidTracker::OutputCharact> track_results =
        std::vector<vitis::ai::ReidTracker::OutputCharact>(
            reid_tracker_->track(image, 1, input_characts, true, true));
  return track_results;

}
} // namespace ai
} // namespace vitis

using namespace cv;

// This function is used to process the multi-posedetect results and show on the image
static cv::Mat process_result_reid_tracker_with_refinedet(
    cv::Mat &image, const ReidTrackerResult &results,
    bool is_jpeg) {
  (void)process_result_reid_tracker_with_refinedet;
  for (auto &r : results) {
    auto box = get<1>(r);
    float x = box.x * (image.cols);
    float y = box.y * (image.rows);
    int xmin = x;
    int ymin = y;
    int xmax = x + (box.width) * (image.cols);
    int ymax = y + (box.height) * (image.rows);
    xmin = std::min(std::max(xmin, 0), image.cols);
    xmax = std::min(std::max(xmax, 0), image.cols);
    ymin = std::min(std::max(ymin, 0), image.rows);
    ymax = std::min(std::max(ymax, 0), image.rows);
    uint64_t gid = get<0>(r);
    cout << "RESULT: "
              << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax
              << "\t" << gid << "\n";
    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  cv::Scalar(255, 0, 0), 2, 1, 0);
    cv::putText(image, to_string(gid), cv::Point(xmin, ymin), 1, 2, cv::Scalar{0, 0, 255},
                    2);
  }
  return image;
}

int main(int argc, char *argv[]) {
  // set the layout
  // assign to Lvalue : static std::vector<cv::Rect> rects, the coordinates of each window 
  gui_layout() = {{0, 0, 960, 540}, {960, 0, 960, 540}, {0, 540, 960, 540}, {960, 540, 960, 540}};
  // assign to Lvalue : set background image
  gui_background() = cv::imread("/usr/share/weston/logo.jpg");
  // init each dpu filter and process instance, using video demo framework
  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv,
      {
       //[] {
       //  return vitis::ai::create_dpu_filter(
       //      [] { return vitis::ai::RefineDetReidTracker::create(); },
       //      process_result_reid_tracker_with_refinedet);
       //},
       //[] {
       //  return vitis::ai::create_dpu_filter(
       //      [] { return vitis::ai::RefineDetReidTracker::create(); },
       //      process_result_reid_tracker_with_refinedet);
       //},
       //[] {
       //  return vitis::ai::create_dpu_filter(
       //      [] { return vitis::ai::RefineDetReidTracker::create(); },
       //      process_result_reid_tracker_with_refinedet);
       //},
       [] {
         return vitis::ai::create_dpu_filter(
             [] { return vitis::ai::RefineDetReidTracker::create(); },
             process_result_reid_tracker_with_refinedet);
       }});
}
