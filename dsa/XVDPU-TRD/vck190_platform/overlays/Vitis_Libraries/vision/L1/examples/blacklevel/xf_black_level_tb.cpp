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

#ifndef __XF_BLACK_LEVEL_TB_CPP__
#define __XF_BLACK_LEVEL_TB_CPP__

#include "common/xf_headers.hpp"
#include "xf_vision_message.h"
#define XF_HLS_MODE 1
#include "xf_black_level_config.h"
#define BLACK_LEVEL 32

#if XF_HLS_MODE
void blackLevelCorrection_accel(ap_uint<IMAGE_PTR_WIDTH>* in_img_ptr,
                                ap_uint<IMAGE_PTR_WIDTH>* out_img_ptr,
                                int black_level,
                                float mul_value,
                                int height,
                                int width);
#endif

int main(int argc, char** argv) {
    XfVisionMessage msg;

    int height, width;

    cv::Mat InImg, OutImg, RefImg;

    int BlackLevel = 32;
#if T_8U
    const int MaxLevel = 255; // Or White Level
#else
    const int MaxLevel = 65535; // Or White Level
#endif
    typedef uint8_t Pixel_t;

    float MulValue = (float)(MaxLevel / (MaxLevel - BlackLevel));
    ; // int((MaxLevel*(1<<IMAGE_MUL_FL_BITS))/(MaxLevel - BlackLevel));

#if T_8U
    InImg = cv::imread(argv[1], 0);
#else
    InImg = cv::imread(argv[1], -1);
#endif
    height = InImg.rows;
    width = InImg.cols;

    printf("%d %d\n", height, width);

    if (argc != 2) return msg.error("Incorrect Usage. Usage: <app.exe> <Input image>");

    // OutImg.create(cv::Size(img.cols, img.rows), img.type());

    RefImg.create(InImg.size(), InImg.type());
    OutImg.create(InImg.size(), InImg.type());

    msg.info("Creating reference for comparision ...");

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            Pixel_t Pixel = InImg.at<Pixel_t>(r, c);
            RefImg.at<Pixel_t>(r, c) = cv::saturate_cast<unsigned char>((Pixel - BlackLevel) * MulValue);
        }
    }

#if XF_HLS_MODE

    blackLevelCorrection_accel((ap_uint<IMAGE_PTR_WIDTH>*)InImg.data, (ap_uint<IMAGE_PTR_WIDTH>*)OutImg.data,
                               BlackLevel, MulValue, height, width);

#else

    msg.info("Registering Kernel ...");

    (void)cl_kernel_mgr::registerKernel("blackLevelCorrection_accel", "krnl_black_level", XCLIN(InImg), XCLOUT(OutImg),
                                        XCLIN(BlackLevel), XCLIN(MulValue), XCLIN(height), XCLIN(width));

    msg.info("Executing HW Kernel ...");
    cl_kernel_mgr::exec_all();

#endif

    msg.info("Dumping reference output ...");
    cv::imwrite("sw_ref_output.png", RefImg);

    msg.info("Dumping Kernel ouptut ...");
    cv::imwrite("hw_output.png", OutImg);

    msg.info("Error checking ...");

    cv::Mat DiffImg;
    float ErrorPercent;

    DiffImg.create(InImg.size(), InImg.type());

    cv::absdiff(RefImg, OutImg, DiffImg);

    msg.info("Dumping difference ...");
    cv::imwrite("sw_ref_hw_diff.png", DiffImg);

    xf::cv::analyzeDiff(DiffImg, 1, ErrorPercent);

    if (ErrorPercent > 10)
        return 1;
    else
        return 0;

    // return XF_APP_SUCCESS;
}

#endif //__XF_BLACK_LEVEL_TB_CPP__
