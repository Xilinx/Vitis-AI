/*
 * Copyright 2021 Xilinx, Inc.
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

/*
 * This is the reference function for L3/examples/cornertracker
 * This function uses the xf_harris reference code for Harris Corner Detection and the OpenCV function
 * calcOpticalFlowPyrLK for Sparse LK Pyramidal Optical Flow (NOT dense)
 * Since the output of the harris corner detector is a list of points it is easier and efficient to use the OpenCV
 * calcOpticalFlowPyrLK function
 * NOTE: The OpenCV calcOpticalFlowPyrLK function does not take NUM_ITERATIONS as an argument
 */

#include "opencv2/video.hpp"
#include "xf_harris_ref.hpp"

#ifndef VIDEO_INPUT
#define VIDEO_INPUT false
#endif

std::vector<cv::Point2f> cornertracker_cv(char* path, int numFrames, int harrisInd) {
    std::vector<cv::Point2f> PointList;
    char img_name[1000], out_img_name[50], pyr_out_img_name[50], pyr_out_img_name2[50];
#if VIDEO_INPUT
    cv::VideoCapture cap;

    std::stringstream imfile;
    imfile.str("");
    imfile << argv[1];
    if (imfile.str() == "") {
        cap.open(0);
        fprintf(stderr, "Invalid input Video\n ");
        if (!cap.isOpened()) {
            fprintf(stderr, "Could not initialize capturing...\n ");
            return 1;
        }
    } else
        cap.open(imfile.str());

    unsigned int imageWidth = (cap.get(CV_CAP_PROP_FRAME_WIDTH));
    unsigned int imageHeight = (cap.get(CV_CAP_PROP_FRAME_HEIGHT));
#else
    cv::Mat frame;
    sprintf(img_name, "%s/im%d.png", path, 0);
    std::cout << "path is" << img_name << std::endl;
    frame = cv::imread(img_name, 1);
    unsigned int imageWidth = frame.cols;
    unsigned int imageHeight = frame.rows;

#endif
    cv::Mat im0(imageHeight, imageWidth, CV_8UC1);
    cv::Mat im1(imageHeight, imageWidth, CV_8UC1);

#if VIDEO_INPUT
    cv::Mat readVideoImage;
    for (int readn = 0; readn < 1; readn++) {
        cap >> readVideoImage;
        if (readVideoImage.empty()) {
            fprintf(stderr, "im1 is empty\n ");
            break;
        }
    }
    cv::cvtColor(readVideoImage, im1, cv::COLOR_BGR2GRAY);
    cv::VideoWriter video("trackedCorners_cv.avi", CV_FOURCC('M', 'J', 'P', 'G'), 5, cv::Size(imageWidth, imageHeight),
                          true);
#else
    cv::cvtColor(frame, im1, cv::COLOR_BGR2GRAY);
#endif
    std::vector<cv::Point2f> p0, p1;
    harris(im1, p0);
    for (int i = 0; i < numFrames; i++) {
        im1.copyTo(im0);
#if VIDEO_INPUT
        cap >> readVideoImage;
        if (readVideoImage.empty()) {
            fprintf(stderr, "im1 is empty\n ");
            break;
        } else
            std::cout << "Read frame no. " << i + 1 << std::endl;
        cv::cvtColor(readVideoImage, im1, cv::COLOR_BGR2GRAY);
#else
        sprintf(img_name, "%s/im%d.png", path, i + 1);
        frame = cv::imread(img_name, 1);
        cv::cvtColor(frame, im1, cv::COLOR_BGR2GRAY);
#endif
        // calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        cv::calcOpticalFlowPyrLK(im0, im1, p0, p1, status, err, cv::Size(WINSIZE_OFLOW, WINSIZE_OFLOW), NUM_LEVELS,
                                 criteria);
        std::vector<cv::Point2f> good_new;
        cv::Mat outputimage;
        cv::cvtColor(im1, outputimage, cv::COLOR_GRAY2BGR);
        for (uint j = 0; j < p0.size(); j++) {
            if (status[j]) {
                good_new.push_back(p1[j]);
                cv::circle(outputimage, p1[j], 2, cv::Scalar(0, 0, 255), -1, 8);
            }
        }
        // update the previous points
        p0.clear();
        if ((i + 1) % harrisInd == 0)
            harris(im1, p0);
        else
            p0 = good_new;

        PointList.clear();
        PointList = good_new;

#if VIDEO_INPUT
        video.write(outputimage);
#else
        sprintf(out_img_name, "out_img%d_cv.png", i);
        cv::imwrite(out_img_name, outputimage);
#endif
    }
    im0.data = NULL;
    im1.data = NULL;
#if VIDEO_INPUT
    cap.release();
    video.release();
#endif
    return PointList;
}
