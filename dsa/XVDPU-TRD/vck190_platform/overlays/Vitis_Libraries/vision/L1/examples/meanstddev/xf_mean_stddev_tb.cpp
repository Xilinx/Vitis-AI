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
#include "xf_mean_stddev_config.h"

template <int CHNLS>
float* xmean(cv::Mat& img) {
    unsigned long Sum[CHNLS] = {0};
    int i, j, c, k;
    float* val = (float*)malloc(CHNLS * sizeof(float));

    /* Sum of All Pixels */
    for (i = 0; i < img.rows; ++i) {
        for (j = 0; j < img.cols; ++j) {
            for (k = 0; k < CHNLS; ++k) {
                if (CHNLS == 1)
                    Sum[k] += img.at<uchar>(i, j); // imag.data[i]}
                else
                    Sum[k] += img.at<cv::Vec3b>(i, j)[k]; // imag.data[i]}
            }
        }
    }
    for (int ch = 0; ch < CHNLS; ++ch) {
        val[ch] = (float)Sum[ch] / (float)(img.rows * img.cols);
    }
    return val;
}
template <int CHNLS>
void variance(cv::Mat& Img, float* mean, double* var) {
    double sum[CHNLS], b_sum = 0.0, g_sum = 0.0, r_sum = 0.0;
    int k;
    double x[CHNLS];
    for (int i = 0; i < Img.rows; i++) {
        for (int j = 0; j < Img.cols; j++) {
            for (k = 0; k < CHNLS; ++k) {
                if (CHNLS == 1)
                    x[k] = (double)mean[k] - ((double)Img.at<uint8_t>(i, j));
                else
                    x[k] = (double)mean[k] - ((double)Img.at<cv::Vec3b>(i, j)[k]);

                sum[k] = sum[k] + pow(x[k], (double)2.0);
            }
        }
    }
    for (int ch = 0; ch < CHNLS; ++ch) {
        var[ch] = (sum[ch] / (double)(Img.rows * Img.cols));
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray;

    // reading in the color image
    in_img = cv::imread(argv[1], 1);
    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }

#if GRAY
    cvtColor(in_img, in_img, cv::COLOR_BGR2GRAY);
#endif

    int channels = in_img.channels();
    printf("Channels - %d\n", channels);
    const int xfcv_channels = XF_CHANNELS(TYPE, __NPPC);
    int height = in_img.rows;
    int width = in_img.cols;

    //////////////// 	Opencv  Reference  ////////////////////////
    float* mean_c = (float*)malloc(channels * sizeof(float));
    float* stddev_c = (float*)malloc(channels * sizeof(float));
    double* var_c = (double*)malloc(channels * sizeof(double));
    float mean_hls[channels], stddev_hls[channels];
    float diff_mean[channels], diff_stddev[channels];

    mean_c = xmean<xfcv_channels>(in_img);
    variance<xfcv_channels>(in_img, mean_c, var_c);

    unsigned short* mean = (unsigned short*)malloc(channels * sizeof(unsigned short));
    unsigned short* stddev = (unsigned short*)malloc(channels * sizeof(unsigned short));

    /// HLS function call
    mean_stddev_accel((ap_uint<PTR_WIDTH>*)in_img.data, mean, stddev, height, width);

    for (int c = 0; c < channels; c++) {
        stddev_c[c] = sqrt(var_c[c]);
        mean_hls[c] = (float)mean[c] / 256;
        stddev_hls[c] = (float)stddev[c] / 256;
        diff_mean[c] = mean_c[c] - mean_hls[c];
        diff_stddev[c] = stddev_c[c] - stddev_hls[c];
        std::cout << "Ref. Mean =" << mean_c[c] << "\t"
                  << "Result =" << mean_hls[c] << "\t"
                  << "ERROR =" << diff_mean[c] << std::endl;
        std::cout << "Ref. Std.Dev. =" << stddev_c[c] << "\t"
                  << "Result =" << stddev_hls[c] << "\t"
                  << "ERROR =" << diff_stddev[c] << std::endl;

        if (abs(diff_mean[c]) > 1 | abs(diff_stddev[c]) > 1) {
            fprintf(stderr, "ERROR: Test Failed.\n ");
            return -1;
        }
    }

    free(mean_c);
    free(stddev_c);
    free(var_c);
    free(mean);
    free(stddev);

    return 0;
}
