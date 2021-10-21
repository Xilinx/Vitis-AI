/*
 * Copyright 2019 Xilinx, Inc.
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

#include "common/xf_headers.hpp"
#include "xf_harris_config.h"
#include "xf_ocv_ref.hpp"

int main(int argc, char** argv) {
    cv::Mat in_img, img_gray;
    cv::Mat hls_out_img, ocv_out_img;
    cv::Mat ocvpnts, hlspnts;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image> \n");
        return -1;
    }
    in_img = cv::imread(argv[1], 0); // reading in the color image

    if (!in_img.data) {
        fprintf(stderr, "Failed to load the image ... %s\n ", argv[1]);
        return -1;
    }

    int Thresh; // Threshold for HLS
    float Th;
    if (FILTER_WIDTH == 3) {
        Th = 30532960.00;
        Thresh = 442;
    } else if (FILTER_WIDTH == 5) {
        Th = 902753878016.0;
        Thresh = 3109;
    } else if (FILTER_WIDTH == 7) {
        Th = 41151168289701888.000000;
        Thresh = 566;
    }
    //	cvtColor(in_img, img_gray, CV_BGR2GRAY);
    // Convert rgb into grayscale
    hls_out_img.create(in_img.rows, in_img.cols, CV_8U); // create memory for hls output image
    ocv_out_img.create(in_img.rows, in_img.cols, CV_8U); // create memory for opencv output image

    ocv_ref(in_img, ocv_out_img, Th);
    /**************		HLS Function	  *****************/
    float K = 0.04;
    int k = K * (1 << 16); // Convert to Q0.16 format
    uint32_t nCorners = 0;
    int imgwidth = in_img.cols;
    int imgheight = in_img.rows;

    cornerHarris_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)hls_out_img.data, imgheight,
                       imgwidth, Thresh, k);

    /// hls_out_img.data = (unsigned char *)imgOutput.copyFrom();
    cv::imwrite("hls_out.jpg", hls_out_img);

    unsigned int val;
    unsigned short int row, col;

    cv::Mat out_img;
    out_img = in_img.clone();

    std::vector<cv::Point> hls_points;
    std::vector<cv::Point> ocv_points;
    std::vector<cv::Point> common_pts;
    /*						Mark HLS points on the image 				*/

    for (int j = 0; j < in_img.rows; j++) {
        int l = 0;
        for (int i = 0; i < (in_img.cols); i++) {
            unsigned char pix = hls_out_img.at<unsigned char>(j, i);
            if (pix != 0) {
                cv::Point tmp;
                tmp.x = i;
                tmp.y = j;
                if ((tmp.x < in_img.cols) && (tmp.y < in_img.rows) && (j > 0)) {
                    hls_points.push_back(tmp);
                }
                short int y, x;
                y = j;
                x = i;
                if (j > 0) cv::circle(out_img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255, 255), 2, 8, 0);
            }
        }
    }

    /*						End of marking HLS points on the image 				*/
    /*						Write HLS and Opencv corners into a file			*/

    ocvpnts = in_img.clone();

    int nhls = hls_points.size();

    /// Drawing a circle around corners
    for (int j = 1; j < ocv_out_img.rows - 1; j++) {
        for (int i = 1; i < ocv_out_img.cols - 1; i++) {
            if ((int)ocv_out_img.at<unsigned char>(j, i)) {
                cv::circle(ocvpnts, cv::Point(i, j), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                ocv_points.push_back(cv::Point(i, j));
            }
        }
    }

    printf("ocv corner count = %d, Hls corner count = %d\n", ocv_points.size(), hls_points.size());
    int nocv = ocv_points.size();

    /*									End
     */
    /*							Find common points in among opencv and HLS
     */
    int ocv_x, ocv_y, hls_x, hls_y;
    for (int j = 0; j < nocv; j++) {
        for (int k = 0; k < nhls; k++) {
            ocv_x = ocv_points[j].x;
            ocv_y = ocv_points[j].y;
            hls_x = hls_points[k].x;
            hls_y = hls_points[k].y;

            if ((ocv_x == hls_x) && (ocv_y == hls_y)) {
                common_pts.push_back(ocv_points[j]);
                break;
            }
        }
    }
    /*							End */
    imwrite("output_hls.png", out_img); // HLS Image
    imwrite("output_ocv.png", ocvpnts); // Opencv Image
    /*						Success, Loss and Gain Percentages */
    float persuccess, perloss, pergain;

    int totalocv = ocv_points.size();
    int ncommon = common_pts.size();
    int totalhls = hls_points.size();
    persuccess = (((float)ncommon / totalhls) * 100);
    perloss = (((float)(totalocv - ncommon) / totalocv) * 100);
    pergain = (((float)(totalhls - ncommon) / totalhls) * 100);

    printf("Commmon = %d\t Success = %f\t Loss = %f\t Gain = %f\n", ncommon, persuccess, perloss, pergain);

    if (persuccess < 60 || totalhls == 0) return 1;

    return 0;
}
