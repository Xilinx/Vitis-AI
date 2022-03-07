/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_blobfromimage_config.h"

#include <sys/time.h>

#include "xcl2.hpp"
#include "xf_opencl_wrap.hpp"

int main(int argc, char* argv[]) {
    struct timeval start_pp_sw, end_pp_sw;
    double lat_pp_sw = 0.0f;
    cv::Mat img_tmp, result_hls, result_ocv, error;

    if (argc != 4) {
        fprintf(stderr,
                "\n Incorrect Usage. Usage is ./.exe <input image> <output "
                "image height> <output image width>");
        return -1;
    }
    img_tmp = cv::imread(argv[1], 1);
    if (!img_tmp.data) {
        fprintf(stderr, "\n image not found");
        return -1;
    }
    int in_width, in_height;
    int out_width, out_height;

    in_width = img_tmp.cols;
    in_height = img_tmp.rows;

    // Add padding in the input image
    int in_pad = in_width;
    int in_stride = in_width + in_pad;
    cv::Mat img;
    cv::copyMakeBorder(img_tmp, img, 0, 0, 0, (in_stride - in_width), cv::BORDER_CONSTANT, 0);
    cv::imwrite("in_img_pad.png", img);

    out_height = atoi(argv[2]);
    out_width = atoi(argv[3]);

    int resize_height = 300;
    int resize_width = 300;

    int roi_posx = 10;
    int roi_posy = 10;

    result_hls.create(cv::Size(out_width, out_height), CV_8UC3);

    // Mean and Scale values
    float params[6];
    params[3] = params[4] = params[5] = 0.25; // scale values
    // Mean values
    params[0] = 128.0f;
    params[1] = 128.0f;
    params[2] = 128.0f;

    /////////////////////////////////////// CL
    /// Wrapper///////////////////////////////////////
    (void)cl_kernel_mgr::registerKernel(
        "blobfromimage_accel", "krnl_blobfromimage_accel", XCLIN(img), XCLOUT(result_hls),
        XCLIN(params, 6 * sizeof(float)), XCLIN(in_width), XCLIN(in_height), XCLIN(in_stride), XCLIN(resize_width),
        XCLIN(resize_height), XCLIN(out_width), XCLIN(out_height), XCLIN(out_width), XCLIN(roi_posx), XCLIN(roi_posy));
    cl_kernel_mgr::exec_all();
    /////////////////////////////////////// end of CL
    //////////////////////////////////////////

    /*Reference Implementation*/

    cv::imwrite("hw_op.png", result_hls); // Write output image from Kernel
    gettimeofday(&start_pp_sw, 0);
    cv::Rect cropROI(0, 0, in_stride - in_width, in_height);
    cv::Mat croppedImage = img(cropROI);                                         // Remove padding from input image
    cv::resize(croppedImage, result_ocv, cv::Size(resize_width, resize_height)); // First Resize the image
    cv::imwrite("cv_resized.png", result_ocv);
    cv::Rect myROI(roi_posx, roi_posy, out_width, out_height);
    cv::Mat result_ocv_crop_orr = result_ocv(myROI);
    cv::Mat result_ocv_crop;
    result_ocv_crop_orr.copyTo(result_ocv_crop); // Need to this to properly access the pixel data
    cv::imwrite("cv_crop.png", result_ocv_crop);
    uchar* img_ocv_data = result_ocv_crop.data;
    int frame_cntr1 = 0;
    float* data_ptr_cv = (float*)malloc(out_width * out_height * 3 * sizeof(float));
    int idx = 0;
    float* dst1_cv = &data_ptr_cv[0];
    float* dst2_cv = &data_ptr_cv[out_height * out_width];
    float* dst3_cv = &data_ptr_cv[(3 - 1) * out_width * out_height];
    for (int ll_rows = 0; ll_rows < out_height; ll_rows++) {
        for (int ll_cols = 0; ll_cols < out_width; ll_cols++) {
            dst1_cv[idx] = (float)(img_ocv_data[frame_cntr1++] - params[0]) * params[3];
            dst2_cv[idx] = (float)(img_ocv_data[frame_cntr1++] - params[1]) * params[4];
            dst3_cv[idx] = (float)(img_ocv_data[frame_cntr1++] - params[2]) * params[5];
            idx++;
        }
    }
    gettimeofday(&end_pp_sw, 0);

    lat_pp_sw = (end_pp_sw.tv_sec * 1e6 + end_pp_sw.tv_usec) - (start_pp_sw.tv_sec * 1e6 + start_pp_sw.tv_usec);
    std::cout << "\n\n Software pre-processing latency " << lat_pp_sw / 1000 << "ms" << std::endl;

    // Error Checking
    int frame_cntr = 0;

    float* data_ptr = (float*)malloc(out_width * out_height * 3 * sizeof(float));
    float* dst1 = &data_ptr[0];
    float* dst2 = &data_ptr[out_width * out_height];
    float* dst3 = &data_ptr[(3 - 1) * out_width * out_height];

    signed char* img_data = (signed char*)result_hls.data;
    float err_th = 1.0;
    float max_error1 = 0.0, max_error2 = 0.0, max_error3 = 0.0;
    int err_cnt1 = 0, err_cnt2 = 0, err_cnt3 = 0;
    int idx1 = 0;
    for (int l_rows = 0; l_rows < out_height; l_rows++) {
        for (int l_cols = 0; l_cols < out_width; l_cols++) {
            dst1[idx1] = (float)img_data[frame_cntr++];
            dst2[idx1] = (float)img_data[frame_cntr++];
            dst3[idx1] = (float)img_data[frame_cntr++];

            float err1 = fabs(dst1[idx1] - dst1_cv[idx1]);
            float err2 = fabs(dst2[idx1] - dst2_cv[idx1]);
            float err3 = fabs(dst3[idx1] - dst3_cv[idx1]);
            if (err1 > err_th && err_cnt1 < 20) {
                std::cout << "Ref = " << dst1_cv[idx1] << "HW = " << dst1[idx1] << std::endl;
                err_cnt1++;
            }
            if (err1 > max_error1) {
                max_error1 = err1;
            }
            if (err2 > max_error2) {
                max_error2 = err2;
            }
            if (err3 > max_error3) {
                max_error3 = err3;
            }
            if (err2 > err_th) {
                err_cnt2++;
            }
            if (err3 > err_th) {
                err_cnt3++;
            }
            idx1++;

        } // l_cols
    }     // l_rows

    std::cout << "\nMax Errors: channel 1 = " << max_error1;
    std::cout << "\tchannel 2 = " << max_error2;
    std::cout << "\tchannel 3 = " << max_error3;
    std::cout << "\nError Counts: Ch1 = " << err_cnt1;
    std::cout << "\tCh2 = " << err_cnt2;
    std::cout << "\tCh3 = " << err_cnt3 << std::endl;

    if (max_error1 > err_th || max_error2 > err_th || max_error3 > err_th) {
        fprintf(stderr, "\n Test Failed\n");
        return -1;

    } else {
        std::cout << "Test Passed " << std::endl;
        return 0;
    }
}
