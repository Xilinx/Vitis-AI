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

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "xf_config_params.h"
#include "../api/xf_tvl1.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef unsigned char pix_t;
typedef unsigned char IN_TYPE2;

typedef struct __rgba {
    IN_TYPE2 r, g, b;
    IN_TYPE2 a; // can be unused
} rgba_t;
typedef struct __rgb { IN_TYPE2 r, g, b; } rgb_t;

static void getPseudoColorInt(pix_t pix, float fx, float fy, rgba_t& rgba) {
    // normalization factor is key for good visualization. Make this auto-ranging
    // or controllable from the host TODO
    // const int normFac = 127/2;
    const int normFac = 10;

    int y = 127 + (int)(fy * normFac);
    int x = 127 + (int)(fx * normFac);
    if (y > 255) y = 255;
    if (y < 0) y = 0;
    if (x > 255) x = 255;
    if (x < 0) x = 0;

    rgb_t rgb;
    if (x > 127) {
        if (y < 128) {
            // 1 quad
            rgb.r = x - 127 + (127 - y) / 2;
            rgb.g = (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 4 quad
            rgb.r = x - 127;
            rgb.g = 0;
            rgb.b = y - 127;
        }
    } else {
        if (y < 128) {
            // 2 quad
            rgb.r = (127 - y) / 2;
            rgb.g = 127 - x + (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 3 quad
            rgb.r = 0;
            rgb.g = 128 - x;
            rgb.b = y - 127;
        }
    }

    rgba.r = pix / 4 + 3 * rgb.r / 4;
    rgba.g = pix / 4 + 3 * rgb.g / 4;
    rgba.b = pix / 4 + 3 * rgb.b / 4;
    rgba.a = 255;
    // rgba.r = rgb.r;
    // rgba.g = rgb.g;
    // rgba.b = rgb.b ;
}

void writeOpticalFlowToFile(const Mat_<Point2f>& flow, const string& fileName)
{
    const char FLO_TAG_STRING[] = "PIEH";    // use this when WRITING the file
    ofstream file(fileName.c_str(), ios_base::binary);

    file << FLO_TAG_STRING;

    file.write((const char*) &flow.cols, sizeof(int));
    file.write((const char*) &flow.rows, sizeof(int));

    for (int i = 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {
            const Point2f u = flow(i, j);

            file.write((const char*) &u.x, sizeof(float));
            file.write((const char*) &u.y, sizeof(float));
        }
    }
}

void Write_FLowimage(cv::Mat frame1, Mat_<Point2f> flow, int id, std::string out_dir)
{

    const string gold_flow_path = "tvl1_flow.flo";
    writeOpticalFlowToFile(flow, gold_flow_path);

    rgba_t pix;
    Point2f u;
    cv::Mat frame_out(frame1.size(), CV_8UC4);

    for (int r = 0; r < frame1.rows; r++) {
        for (int c = 0; c < frame1.cols; c++) {
            u = flow(r, c);

            getPseudoColorInt(frame1.at<pix_t>(r, c), u.x, u.y, pix);

            frame_out.at<unsigned int>(r, c) = ((unsigned int)pix.a << 24 |
                                                (unsigned int)pix.b << 16 |
                                                (unsigned int)pix.g << 8 |
                                                (unsigned int)pix.r);
        }
    }

    string out_img = "/flow_out_" + std::to_string(id) + ".png";    
    cv::imwrite(out_dir+out_img, frame_out);
    //std::cout << "flow img " << out_dir+out_img << std::endl;
    //cv::imwrite(out_img, frame_out);
}

