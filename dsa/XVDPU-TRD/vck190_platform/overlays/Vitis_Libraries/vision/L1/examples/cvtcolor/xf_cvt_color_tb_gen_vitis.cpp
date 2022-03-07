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
#define ERROR_THRESHOLD 2

#include "xf_cvt_color_config_gen_vitis.h"

int main(int argc, char** argv) {
    cv::Mat imgInput0, imgInput1, imgInput2;
    cv::Mat refOutput0, refOutput1, refOutput2;
    cv::Mat errImg0, errImg1, errImg2;

#if RGBA2IYUV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGBA);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_rgba2iyuv(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if RGBA2NV12
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGBA);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_rgba2nv12(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if RGBA2NV21
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGBA);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_rgba2nv21(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if RGBA2YUV4
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGBA);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput2(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_rgba2yuv4(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, HEIGHT);
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, HEIGHT);
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if RGB2IYUV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_rgb2iyuv(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if RGB2NV12
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_rgb2nv12(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if RGB2NV21
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_rgb2nv21(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if RGB2YUV4
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput2(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_rgb2yuv4(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, HEIGHT);
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, HEIGHT);
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if RGB2UYVY
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_rgb2uyvy(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_UYVY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_UYVY.png", imgOutput0);
#endif
#if RGB2YUYV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_rgb2yuyv(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YUYV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YUYV.png", imgOutput0);
#endif
#if RGB2BGR
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_rgb2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if BGR2UYVY
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_bgr2uyvy(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_UYVY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_UYVY.png", imgOutput0);
#endif
#if BGR2YUYV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_bgr2yuyv(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YUYV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YUYV.png", imgOutput0);
#endif
#if BGR2RGB
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_bgr2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if BGR2NV12
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_bgr2nv12(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if BGR2NV21
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_bgr2nv21(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if IYUV2NV12
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], 0);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput1 = (ap_uint<8 * NPC1>*)imgInput1.data;

    imgInput2 = cv::imread(argv[3], 0);
    if (!imgInput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput2 = (ap_uint<8 * NPC1>*)imgInput2.data;

    refOutput0 = cv::imread(argv[4], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[5], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_iyuv2nv12(_imgInput0, _imgInput1, _imgInput2, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if IYUV2RGBA
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], 0);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput1 = (ap_uint<8 * NPC1>*)imgInput1.data;

    imgInput2 = cv::imread(argv[3], 0);
    if (!imgInput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput2 = (ap_uint<8 * NPC1>*)imgInput2.data;

    refOutput0 = cv::imread(argv[4], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC4);

    cvtcolor_iyuv2rgba(_imgInput0, _imgInput1, _imgInput2, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGBA2BGR);

    cv::imwrite("out_RGBA.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC4);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGBA.png", imgOutput0);
#endif
#if IYUV2RGB
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], 0);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput1 = (ap_uint<8 * NPC1>*)imgInput1.data;

    imgInput2 = cv::imread(argv[3], 0);
    if (!imgInput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput2 = (ap_uint<8 * NPC1>*)imgInput2.data;

    refOutput0 = cv::imread(argv[4], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_iyuv2rgb(_imgInput0, _imgInput1, _imgInput2, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if IYUV2YUV4
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], 0);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput1 = (ap_uint<8 * NPC1>*)imgInput1.data;

    imgInput2 = cv::imread(argv[3], 0);
    if (!imgInput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput2 = (ap_uint<8 * NPC1>*)imgInput2.data;

    refOutput0 = cv::imread(argv[4], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[5], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput2 = cv::imread(argv[6], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[6]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput2(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_iyuv2yuv4(_imgInput0, _imgInput1, _imgInput2, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, HEIGHT);
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, HEIGHT);
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if NV122IYUV
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[5], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_nv122iyuv(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if NV122RGBA
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC4);

    cvtcolor_nv122rgba(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGBA2BGR);

    cv::imwrite("out_RGBA.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC4);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGBA.png", imgOutput0);
#endif
#if NV122YUV4
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput2 = cv::imread(argv[5], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput2(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_nv122yuv4(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, HEIGHT);
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, HEIGHT);
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if NV122RGB
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_nv122rgb(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if NV122BGR
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_nv122bgr(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if NV122UYVY
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_nv122uyvy(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_UYVY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_UYVY.png", imgOutput0);
#endif
#if NV122YUYV
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_nv122yuyv(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YUYV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YUYV.png", imgOutput0);
#endif
#if NV122NV21
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_nv122nv21(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if NV212IYUV
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[5], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_nv212iyuv(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if NV212RGBA
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC4);

    cvtcolor_nv212rgba(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGBA2BGR);

    cv::imwrite("out_RGBA.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC4);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGBA.png", imgOutput0);
#endif
#if NV212RGB
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_nv212rgb(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if NV212BGR
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_nv212bgr(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if NV212YUV4
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput2 = cv::imread(argv[5], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[5]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput2(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_nv212yuv4(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, HEIGHT);
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, HEIGHT);
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if NV212UYVY
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_nv212uyvy(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_UYVY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_UYVY.png", imgOutput0);
#endif
#if NV212YUYV
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_nv212yuyv(_imgInput0, _imgInput1, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YUYV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YUYV.png", imgOutput0);
#endif
#if NV212NV12
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    imgInput1 = cv::imread(argv[2], -1);
    if (!imgInput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgInput1 = (ap_uint<16 * NPC2>*)imgInput1.data;

    refOutput0 = cv::imread(argv[3], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[4], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_nv212nv12(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if UYVY2IYUV
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_uyvy2iyuv(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if UYVY2NV12
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_uyvy2nv12(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if UYVY2NV21
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_uyvy2nv21(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if UYVY2RGBA
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC4);

    cvtcolor_uyvy2rgba(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGBA2BGR);

    cv::imwrite("out_RGBA.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC4);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGBA.png", imgOutput0);
#endif
#if UYVY2RGB
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_uyvy2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if UYVY2BGR
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_uyvy2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if UYVY2YUYV
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_uyvy2yuyv(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YUYV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YUYV.png", imgOutput0);
#endif
#if YUYV2IYUV
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], 0);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput1 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    refOutput2 = cv::imread(argv[4], 0);
    if (!refOutput2.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[4]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput2 = (ap_uint<8 * NPC1>*)malloc((HEIGHT / 4) * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 4), WIDTH, CV_8UC1);
    cv::Mat imgOutput2((HEIGHT / 4), WIDTH, CV_8UC1);

    cvtcolor_yuyv2iyuv(_imgInput0, _imgOutput0, _imgOutput1, _imgOutput2);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;
    imgOutput2.data = (unsigned char*)_imgOutput2;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_U.png", imgOutput1);
    cv::imwrite("out_V.png", imgOutput2);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1(WIDTH, (HEIGHT / 4));
    errImg1.create(S1, CV_8UC1);
    cv::Size S2(WIDTH, (HEIGHT / 4));
    errImg2.create(S2, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);
    cv::absdiff(refOutput2, imgOutput2, errImg2);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_U.png", imgOutput1);
    cv::imwrite("err_V.png", imgOutput2);
#endif
#if YUYV2NV12
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_yuyv2nv12(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if YUYV2NV21
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    refOutput1 = cv::imread(argv[3], -1);
    if (!refOutput1.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[3]);
        return -1;
    }
    ap_uint<16 * NPC2>* _imgOutput1 = (ap_uint<16 * NPC2>*)malloc((HEIGHT / 2) * (WIDTH / 2) * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat imgOutput1((HEIGHT / 2), (WIDTH / 2), CV_16UC1);

    cvtcolor_yuyv2nv21(_imgInput0, _imgOutput0, _imgOutput1);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    imgOutput1.data = (unsigned char*)_imgOutput1;

    cv::imwrite("out_Y.png", imgOutput0);
    cv::imwrite("out_UV.png", imgOutput1);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);
    cv::Size S1((WIDTH / 2), (HEIGHT / 2));
    errImg1.create(S1, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);
    cv::absdiff(refOutput1, imgOutput1, errImg1);

    cv::imwrite("err_Y.png", imgOutput0);
    cv::imwrite("err_UV.png", imgOutput1);
#endif
#if YUYV2RGBA
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC4);

    cvtcolor_yuyv2rgba(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGBA2BGR);

    cv::imwrite("out_RGBA.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC4);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGBA.png", imgOutput0);
#endif
#if YUYV2RGB
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_yuyv2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if YUYV2BGR
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_yuyv2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if YUYV2UYVY
    imgInput0 = cv::imread(argv[1], -1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgInput0 = (ap_uint<16 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], -1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<16 * NPC1>* _imgOutput0 = (ap_uint<16 * NPC1>*)malloc(HEIGHT * WIDTH * 16);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_16UC1);

    cvtcolor_yuyv2uyvy(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_UYVY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_16UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_UYVY.png", imgOutput0);
#endif
#if RGB2GRAY
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    cvtColor(imgInput0, imgInput0, cv::COLOR_BGR2RGB);
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_rgb2gray(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_GRAY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_GRAY.png", imgOutput0);
#endif
#if BGR2GRAY
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 0);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgOutput0 = (ap_uint<8 * NPC1>*)malloc(HEIGHT * WIDTH * 8);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC1);

    cvtcolor_bgr2gray(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_GRAY.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC1);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_GRAY.png", imgOutput0);
#endif
#if GRAY2RGB
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_gray2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if GRAY2BGR
    imgInput0 = cv::imread(argv[1], 0);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<8 * NPC1>* _imgInput0 = (ap_uint<8 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_gray2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if RGB2XYZ
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_rgb2xyz(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_XYZ.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_XYZ.png", imgOutput0);
#endif
#if BGR2XYZ
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_bgr2xyz(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_XYZ.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_XYZ.png", imgOutput0);
#endif
#if XYZ2RGB
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_xyz2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if XYZ2BGR
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_xyz2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if RGB2YCrCb
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_rgb2ycrcb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YCrCb.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YCrCb.png", imgOutput0);
#endif
#if BGR2YCrCb
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_bgr2ycrcb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_YCrCb.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_YCrCb.png", imgOutput0);
#endif
#if YCrCb2RGB
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_ycrcb2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if YCrCb2BGR
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_ycrcb2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if RGB2HLS
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_rgb2hls(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_HLS.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_HLS.png", imgOutput0);
#endif
#if BGR2HLS
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_bgr2hls(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_HLS.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_HLS.png", imgOutput0);
#endif
#if HLS2RGB
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_hls2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if HLS2BGR
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_hls2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif
#if RGB2HSV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_rgb2hsv(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_HSV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_HSV.png", imgOutput0);
#endif
#if BGR2HSV
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_bgr2hsv(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_HSV.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_HSV.png", imgOutput0);
#endif
#if HSV2RGB
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_hsv2rgb(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;
    cvtColor(imgOutput0, imgOutput0, cv::COLOR_RGB2BGR);

    cv::imwrite("out_RGB.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_RGB.png", imgOutput0);
#endif
#if HSV2BGR
    imgInput0 = cv::imread(argv[1], 1);
    if (!imgInput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[1]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgInput0 = (ap_uint<32 * NPC1>*)imgInput0.data;

    refOutput0 = cv::imread(argv[2], 1);
    if (!refOutput0.data) {
        fprintf(stderr, "Can't open image %s !!.\n ", argv[2]);
        return -1;
    }
    ap_uint<32 * NPC1>* _imgOutput0 = (ap_uint<32 * NPC1>*)malloc(HEIGHT * WIDTH * 32);

    cv::Mat imgOutput0(HEIGHT, WIDTH, CV_8UC3);

    cvtcolor_hsv2bgr(_imgInput0, _imgOutput0);

    imgOutput0.data = (unsigned char*)_imgOutput0;

    cv::imwrite("out_BGR.png", imgOutput0);
    cv::Size S0(WIDTH, HEIGHT);
    errImg0.create(S0, CV_8UC3);

    cv::absdiff(refOutput0, imgOutput0, errImg0);

    cv::imwrite("err_BGR.png", imgOutput0);
#endif

    float err_per;
    xf::cv::analyzeDiff(errImg0, ERROR_THRESHOLD, err_per);

    if (err_per > 3.0f) {
        fprintf(stderr, "\n1st Image Test Failed\n");
    }
#if (IYUV2NV12 || RGBA2NV12 || RGBA2NV21 || UYVY2NV12 || YUYV2NV12 || NV122IYUV || NV212IYUV || IYUV2YUV4 || \
     NV122YUV4 || NV212YUV4 || RGBA2IYUV || RGBA2YUV4 || UYVY2IYUV || YUYV2IYUV || NV122NV21 || NV212NV12)
    xf::cv::analyzeDiff(errImg1, ERROR_THRESHOLD, err_per);
    if (err_per > 3.0f) {
        fprintf(stderr, "\n2nd Image Test Failed\n");
        return 1;
    }

#endif
#if (IYUV2YUV4 || NV122IYUV || NV122YUV4 || NV212IYUV || NV212YUV4 || RGBA2IYUV || RGBA2YUV4 || UYVY2IYUV || YUYV2IYUV)
    xf::cv::analyzeDiff(errImg2, ERROR_THRESHOLD, err_per);
    if (err_per > 3.0f) {
        fprintf(stderr, "\n3rd Image Test Failed\n");
        return 1;
    }
#endif
    /* ## *************************************************************** ##*/
    return 0;
}
