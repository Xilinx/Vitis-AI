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

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_dense_npyr_optical_flow_config.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define HLS 0

#if !HLS

#include "xcl2.hpp"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
#include <string>

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

static void getOutPix(float* fx, float* fy, pix_t* p, hls::stream<rgba_t>& out_pix, int rows, int cols, int size) {
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE
            // clang-format on
            float fx_ = *(fx + r * cols + c);
            float fy_ = *(fy + r * cols + c);

            pix_t p_ = *(p + r * cols + c);
            rgba_t out_pix_;
            getPseudoColorInt(p_, fx_, fy_, out_pix_);

            out_pix.write(out_pix_);
        }
    }
}

static void writeMatRowsRGBA(hls::stream<rgba_t>& pixStream, unsigned int* dst, int rows, int cols, int size) {
    for (int i = 0; i < size; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS*COLS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        rgba_t tmpData = pixStream.read();
        *(dst + i) = (unsigned int)tmpData.a << 24 | (unsigned int)tmpData.b << 16 | (unsigned int)tmpData.g << 8 |
                     (unsigned int)tmpData.r;
    }
}

void dense_non_pyr_of_accel(ap_uint<INPUT_PTR_WIDTH>* img_curr,
                            ap_uint<INPUT_PTR_WIDTH>* img_prev,
                            float* img_outx,
                            float* img_outy,
                            int cols,
                            int rows);

int main(int argc, char** argv) {
    cv::Mat frame0, frame1;
#if !HLS
    cv::Mat flowx, flowy;
#endif
    cv::Mat frame_out;

    if (argc != 3) {
        fprintf(stderr, "Usage incorrect. Correct usage: ./exe <current frame> <next frame>\n");
        return -1;
    }

    frame0 = cv::imread(argv[1], 0);
    frame1 = cv::imread(argv[2], 0);

    if (frame0.empty() || frame1.empty()) {
        fprintf(stderr, "input files not found!\n");
        return -1;
    }

    frame_out.create(frame0.rows, frame0.cols, CV_8UC4);
#if !HLS
    flowx.create(frame0.rows, frame0.cols, CV_32FC1);
    flowy.create(frame0.rows, frame0.cols, CV_32FC1);
#endif

    int cnt = 0;
    unsigned char p1, p2, p3, p4;
    unsigned int pix = 0;

    char out_string[200];

#if HLS
    static xf::cv::Mat<XF_32FC1, MAX_HEIGHT, MAX_WIDTH, OF_PIX_PER_CLOCK> flowx(frame0.rows, frame0.cols);
    static xf::cv::Mat<XF_32FC1, MAX_HEIGHT, MAX_WIDTH, OF_PIX_PER_CLOCK> flowy(frame0.rows, frame0.cols);

#endif
    /////////////////////////////////////// CL ////////////////////////

    int height = frame0.rows;
    int width = frame0.cols;
#if HLS
    dense_non_pyr_of_accel(frame0.data, frame1.data, (float*)flowx.data, (float*)flowy.data, height, width);
#endif
#if !HLS

    std::cout << "Starting xrt programmingms" << std::endl;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    std::cout << "device context created" << std::endl;

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::cout << "command queue created" << std::endl;

    // Create Program and Kernel
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_lknpyrof");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "dense_non_pyr_of_accel");

    std::cout << "kernel loaded" << std::endl;

    // Allocate Buffer in Global Memory
    cl::Buffer currImageToDevice(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer prevImageToDevice(context, CL_MEM_READ_ONLY, (height * width));

    std::cout << "input buffer created" << std::endl;

    cl::Buffer outxImageFromDevice(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (height * width * OUT_BYTES_PER_CHANNEL), flowx.data);
    cl::Buffer outyImageFromDevice(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (height * width * OUT_BYTES_PER_CHANNEL), flowy.data);

    std::vector<cl::Memory> outBufVec0, outBufVec1;
    outBufVec0.push_back(outxImageFromDevice);
    outBufVec1.push_back(outyImageFromDevice);

    std::cout << "output buffer created" << std::endl;

    krnl.setArg(0, currImageToDevice);
    krnl.setArg(1, prevImageToDevice);
    krnl.setArg(2, outxImageFromDevice);
    krnl.setArg(3, outyImageFromDevice);
    krnl.setArg(4, height);
    krnl.setArg(5, width);

    std::cout << "arguments copied" << std::endl;

    // Copying input data to Device buffer from host memory
    q.enqueueWriteBuffer(currImageToDevice, CL_TRUE, 0, (height * width), frame0.data);
    q.enqueueWriteBuffer(prevImageToDevice, CL_TRUE, 0, (height * width), frame1.data);

    std::cout << "input buffer copied" << std::endl;

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    // Profiling
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    std::cout << "ip returned" << std::endl;
    // Copying Device result data to Host memory
    q.enqueueMigrateMemObjects(outBufVec0, CL_MIGRATE_MEM_OBJECT_HOST);
    std::cout << "output x  buffer read" << std::endl;
    q.enqueueMigrateMemObjects(outBufVec1, CL_MIGRATE_MEM_OBJECT_HOST);
    std::cout << "output y  buffer read" << std::endl;

    std::cout << "done" << std::endl;
    q.finish();

#endif
    /////////////////////////////////////// end of CL ////////////////////////

    float* flowx_copy;
    float* flowy_copy;

    flowx_copy = (float*)malloc(MAX_HEIGHT * MAX_WIDTH * (sizeof(float)));
    if (flowx_copy == NULL) {
        fprintf(stderr, "\nFailed to allocate memory for flowx_copy\n");
    }
    flowy_copy = (float*)malloc(MAX_HEIGHT * MAX_WIDTH * (sizeof(float)));
    if (flowy_copy == NULL) {
        fprintf(stderr, "\nFailed to allocate memory for flowy_copy\n");
    }

#if !HLS
    int size = height * width;
    for (int f = 0; f < height; f++) {
        for (int i = 0; i < width; i++) {
            flowx_copy[f * width + i] = flowx.at<float>(f, i);
            flowy_copy[f * width + i] = flowy.at<float>(f, i);
        }
    }
#endif

    unsigned int* outputBuffer;
    outputBuffer = (unsigned int*)malloc(MAX_HEIGHT * MAX_WIDTH * (sizeof(unsigned int)));
    if (outputBuffer == NULL) {
        fprintf(stderr, "\nFailed to allocate memory for outputBuffer\n");
    }

    hls::stream<rgba_t> out_pix("Color pixel");

    getOutPix(flowx_copy, flowy_copy, frame1.data, out_pix, frame0.rows, frame0.cols, frame0.cols * frame0.rows);

    writeMatRowsRGBA(out_pix, outputBuffer, frame0.rows, frame0.cols, frame0.cols * frame0.rows);

    rgba_t* outbuf_copy;
    for (int i = 0; i < frame0.rows; i++) {
        for (int j = 0; j < frame0.cols; j++) {
            outbuf_copy = (rgba_t*)(outputBuffer + i * (frame0.cols) + j);
            p1 = outbuf_copy->r;
            p2 = outbuf_copy->g;
            p3 = outbuf_copy->b;
            p4 = outbuf_copy->a;
            pix = ((unsigned int)p4 << 24) | ((unsigned int)p3 << 16) | ((unsigned int)p2 << 8) | (unsigned int)p1;
            frame_out.at<unsigned int>(i, j) = pix;
        }
    }

    sprintf(out_string, "out_%d.png", cnt);
    cv::imwrite(out_string, frame_out);
    return 0;
}
