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
#include "xf_cvt_color_config.h"

#include "xcl2.hpp"

#define ERROR_THRESHOLD 2

int main(int argc, char** argv) {
    uint16_t img_width;
    uint16_t img_height;

    cv::Mat inputimg0, inputimg1, inputimg2, inputimg;
    cv::Mat outputimg0, outputimg1, outputimg2;
    cv::Mat error_img0, error_img1, error_img2;
    cv::Mat refimage, refimg0, refimg1, refimg2;
    cv::Mat refimage0, refimage1, refimage2;

    cv::Mat img;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_cvt_color");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

#if __SDSCC__
    perf_counter hw_ctr;
#endif
#if IYUV2NV12

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg1 = cv::imread(argv[2], 0);
    if (!inputimg1.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg2 = cv::imread(argv[3], 0);
    if (!inputimg2.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;

    int newwidth_uv = inputimg1.cols / 2;
    int newheight_uv = inputimg1.rows + inputimg2.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;
    int height_u = inputimg1.rows;
    int width_u = inputimg1.cols;
    int height_v = inputimg2.rows;
    int width_v = inputimg2.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_iyuv2nv12", &err));
    printf("started buffer creation task\n");

    OCL_CHECK(err, cl::Buffer imageToDeviceY(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceU(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceV(context, CL_MEM_READ_ONLY, (inputimg2.rows * inputimg2.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceY(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceUV(context, CL_MEM_WRITE_ONLY, (newheight_uv * newwidth_uv * 2), NULL, &err));
    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceY, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceU, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols), inputimg1.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceV, CL_TRUE, 0, (inputimg2.rows * inputimg2.cols), inputimg2.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceY));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceU));
    OCL_CHECK(err, err = krnl.setArg(2, imageToDeviceV));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceY));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDeviceUV));

    OCL_CHECK(err, err = krnl.setArg(5, height));
    OCL_CHECK(err, err = krnl.setArg(6, width));
    OCL_CHECK(err, err = krnl.setArg(7, height_u));
    OCL_CHECK(err, err = krnl.setArg(8, width_u));
    OCL_CHECK(err, err = krnl.setArg(9, height_v));
    OCL_CHECK(err, err = krnl.setArg(10, width_v));
    OCL_CHECK(err, err = krnl.setArg(11, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(12, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceY, CL_TRUE, 0, (newheight_y * newwidth_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceUV, CL_TRUE, 0, (newheight_uv * newwidth_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    refimage0 = cv::imread(argv[4], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[5], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);
    cv::imwrite("y_error.png", error_img0);
    cv::imwrite("UV_error.png", error_img1);
#endif
#if IYUV2RGBA

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg1 = cv::imread(argv[2], 0);
    if (!inputimg1.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg2 = cv::imread(argv[3], 0);
    if (!inputimg2.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }

    cv::Size S0(inputimg0.cols, inputimg0.rows);
    outputimg0.create(S0, CV_8UC4);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows; // + inputimg1.rows + inputimg2.rows;

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S(newwidth, newheight);
    outputimg1.create(S, CV_8UC3);

    // outputimg_ocv.create(S,CV_8UC4);
    error_img0.create(S, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;
    int height_u = inputimg1.rows;
    int width_u = inputimg1.cols;
    int height_v = inputimg2.rows;
    int width_v = inputimg2.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_iyuv2rgba", &err));

    OCL_CHECK(err, cl::Buffer imageToDeviceY(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceU(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceV(context, CL_MEM_READ_ONLY, (inputimg2.rows * inputimg2.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergba(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 4), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceY, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceU, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols), inputimg1.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceV, CL_TRUE, 0, (inputimg2.rows * inputimg2.cols), inputimg2.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceY));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceU));
    OCL_CHECK(err, err = krnl.setArg(2, imageToDeviceV));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicergba));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, height_u));
    OCL_CHECK(err, err = krnl.setArg(7, width_u));
    OCL_CHECK(err, err = krnl.setArg(8, height_v));
    OCL_CHECK(err, err = krnl.setArg(9, width_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergba, CL_TRUE, 0, (newwidth * newheight * 4),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    cvtColor(outputimg0, outputimg1, cv::COLOR_RGBA2BGR);
    cv::imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[4], 1);

    absdiff(outputimg1, refimage, error_img0);

#endif
#if IYUV2RGB

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg1 = cv::imread(argv[2], 0);
    if (!inputimg1.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg2 = cv::imread(argv[3], 0);
    if (!inputimg2.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }

    cv::Size S0(inputimg0.cols, inputimg0.rows);
    outputimg0.create(S0, CV_8UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows; // + inputimg1.rows + inputimg2.rows;

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S(newwidth, newheight);
    outputimg0.create(S, CV_8UC3);

    // outputimg_ocv.create(S,CV_8UC4);
    error_img0.create(S, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;
    int height_u = inputimg1.rows;
    int width_u = inputimg1.cols;
    int height_v = inputimg2.rows;
    int width_v = inputimg2.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_iyuv2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDeviceY(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceU(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceV(context, CL_MEM_READ_ONLY, (inputimg2.rows * inputimg2.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * OUTPUT_CH_TYPE),
                                                 NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceY, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceU, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols), inputimg1.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceV, CL_TRUE, 0, (inputimg2.rows * inputimg2.cols), inputimg2.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceY));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceU));
    OCL_CHECK(err, err = krnl.setArg(2, imageToDeviceV));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, height_u));
    OCL_CHECK(err, err = krnl.setArg(7, width_u));
    OCL_CHECK(err, err = krnl.setArg(8, height_v));
    OCL_CHECK(err, err = krnl.setArg(9, width_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (newwidth * newheight * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////
    cvtColor(outputimg0, outputimg0, cv::COLOR_RGB2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[4], 1);

    absdiff(outputimg0, refimage, error_img0);
    cv::imwrite("diff.jpg", error_img0);

#endif
#if IYUV2YUV4

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg1 = cv::imread(argv[2], 0);
    if (!inputimg1.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }
    inputimg2 = cv::imread(argv[3], 0);
    if (!inputimg2.data) {
        fprintf(stderr, "Can't open image !!\n ");
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_u = inputimg1.cols;
    int newheight_u = inputimg1.rows << 2;
    int newwidth_v = inputimg2.cols;
    int newheight_v = inputimg2.rows << 2;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u, newheight_u);
    outputimg1.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);

    cv::Size S2(newwidth_v, newheight_v);
    outputimg2.create(S2, CV_8UC1);
    error_img2.create(S2, CV_8UC1);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;
    int height_u = inputimg1.rows;
    int width_u = inputimg1.cols;
    int height_v = inputimg2.rows;
    int width_v = inputimg2.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_iyuv2yuv4", &err));

    OCL_CHECK(err, cl::Buffer imageToDeviceY(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceU(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceV(context, CL_MEM_READ_ONLY, (inputimg2.rows * inputimg2.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceY(context, CL_MEM_READ_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceU(context, CL_MEM_READ_ONLY, (newheight_u * newwidth_u), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_READ_ONLY, (newheight_v * newwidth_v), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceY, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceU, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols), inputimg1.data));
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDeviceV, CL_TRUE, 0, (inputimg2.rows * inputimg2.cols), inputimg2.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceY));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceU));
    OCL_CHECK(err, err = krnl.setArg(2, imageToDeviceV));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceY));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDeviceU));
    OCL_CHECK(err, err = krnl.setArg(5, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(6, height));
    OCL_CHECK(err, err = krnl.setArg(7, width));
    OCL_CHECK(err, err = krnl.setArg(8, height_u));
    OCL_CHECK(err, err = krnl.setArg(9, width_u));
    OCL_CHECK(err, err = krnl.setArg(10, height_v));
    OCL_CHECK(err, err = krnl.setArg(11, width_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceY, CL_TRUE, 0, (newheight_y * newwidth_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceU, CL_TRUE, 0, (newheight_u * newwidth_u),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newheight_v * newwidth_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    refimage0 = cv::imread(argv[4], 0);
    if (!refimage0.data) {
        fprintf(stderr, "unable to open image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[5], 0);
    if (!refimage1.data) {
        fprintf(stderr, "unable to open image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[6], 0);
    if (!refimage2.data) {
        fprintf(stderr, "unable to open image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

    imwrite("error_u.png", error_img1);
    imwrite("error_V.png", error_img2);

#endif
#if NV122IYUV

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;

    int newwidth_u_v = inputimg1.cols << 1;
    int newheight_u_v = inputimg1.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);

    outputimg2.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122iyuv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(5, height_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_y));
    OCL_CHECK(err, err = krnl.setArg(7, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(8, width_u_y));
    OCL_CHECK(err, err = krnl.setArg(9, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(10, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(11, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(12, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimg0 = cv::imread(argv[3], 0);
    if (!refimg0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimg1 = cv::imread(argv[4], 0);
    if (!refimg1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimg2 = cv::imread(argv[5], 0);
    if (!refimg2.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }

    cv::absdiff(refimg0, outputimg0, error_img0);
    cv::absdiff(refimg1, outputimg1, error_img1);
    cv::absdiff(refimg2, outputimg2, error_img2);

    imwrite("error_Y.png", error_img0);
    imwrite("error_U.png", error_img1);
    imwrite("error_V.png", error_img2);
#endif

#if NV122RGBA

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC4);
    error_img0.create(S0, CV_8UC4);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122rgba", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergba(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 4), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergba));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergba, CL_TRUE, 0, (newwidth * newheight * 4),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, CV_RGBA2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg0, refimage, error_img0);

#endif
#if NV122RGB

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);
    error_img0.create(S0, CV_8UC3);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 3), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (newwidth * newheight * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, cv::COLOR_RGB2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);
#endif

#if NV122BGR

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);
    error_img0.create(S0, CV_8UC3);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 3), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (newwidth * newheight * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);
#endif
#if NV212BGR

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);
    error_img0.create(S0, CV_8UC3);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 3), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (newwidth * newheight * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);
#endif

#if NV122UYVY

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122uyvy", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDeviceuyvy(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuyvy, CL_TRUE, 0, (newwidth * newheight * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    imwrite("out_uyvy.png", outputimg0);

    refimage = cv::imread(argv[3], -1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);

#endif

#if NV122NV21

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;
    int newwidth_uv = inputimg1.cols;
    int newheight_uv = inputimg1.rows;

    cv::Size S0(newwidth, newheight);
    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg0.create(S0, CV_8UC1);
    outputimg1.create(S1, CV_16UC1);
    error_img0.create(S0, CV_8UC1);
    error_img1.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122nv21", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err,
              cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL,
                                                &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(4, height_y));
    OCL_CHECK(err, err = krnl.setArg(5, width_y));
    OCL_CHECK(err, err = krnl.setArg(6, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(7, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    refimage0 = cv::imread(argv[3], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[4], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open VU reference image\n ");
        return (1);
    }
    absdiff(refimage1, outputimg1, error_img1);
    absdiff(refimage0, outputimg0, error_img0);
    cv::imwrite("error_img0.png", error_img0);
    cv::imwrite("error_img1.png", error_img1);
#endif
#if NV212NV12

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;
    int newwidth_uv = inputimg1.cols;
    int newheight_uv = inputimg1.rows;

    cv::Size S0(newwidth, newheight);
    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg0.create(S0, CV_8UC1);
    outputimg1.create(S1, CV_16UC1);
    error_img0.create(S0, CV_8UC1);
    error_img1.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212nv12", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err,
              cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL,
                                                &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(4, height_y));
    OCL_CHECK(err, err = krnl.setArg(5, width_y));
    OCL_CHECK(err, err = krnl.setArg(6, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(7, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    refimage0 = cv::imread(argv[3], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[4], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open VU reference image\n ");
        return (1);
    }
    absdiff(refimage1, outputimg1, error_img1);
    absdiff(refimage0, outputimg0, error_img0);
    cv::imwrite("error_img0.png", error_img0);
    cv::imwrite("error_img1.png", error_img1);
#endif
#if NV122YUYV

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122yuyv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth * newheight * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    imwrite("out_yuyv.png", outputimg0);

    refimage = cv::imread(argv[3], -1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);

#endif
#if NV212UYVY

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212uyvy", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDeviceuyvy(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuyvy, CL_TRUE, 0, (newwidth * newheight * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    imwrite("out_uyvy.png", outputimg0);

    refimage = cv::imread(argv[3], -1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);

#endif
#if NV212YUYV

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212yuyv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth * newheight * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    imwrite("out_yuyv.png", outputimg0);

    refimage = cv::imread(argv[3], -1);

    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);

#endif
#if NV122YUV4

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;

    int newwidth_u_v = inputimg1.cols << 1;
    int newheight_u_v = inputimg1.rows << 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);

    outputimg2.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    img_width = inputimg0.cols;
    img_height = inputimg0.rows; // + inputimg1.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv122yuv4", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(5, height_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_y));
    OCL_CHECK(err, err = krnl.setArg(7, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(8, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[3], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[4], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[5], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif
#if NV212IYUV

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;

    int newwidth_u_v = inputimg1.cols << 1;
    int newheight_u_v = inputimg1.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);

    outputimg2.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212iyuv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(5, height_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_y));
    OCL_CHECK(err, err = krnl.setArg(7, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(8, width_u_y));
    OCL_CHECK(err, err = krnl.setArg(9, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(10, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(11, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(12, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[3], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[4], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[5], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif

#if NV212RGBA

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC4);
    outputimg1.create(S0, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212rgba", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergba(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * 4), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergba));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergba, CL_TRUE, 0, (newwidth * newheight * 4),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg1, CV_RGBA2BGR);
    imwrite("out.png", outputimg1);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg1, refimage, error_img0);

#endif
#if NV212RGB

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth = inputimg0.cols;
    int newheight = inputimg0.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (newwidth * newheight * OUTPUT_CH_TYPE),
                                                 NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(3, height_y));
    OCL_CHECK(err, err = krnl.setArg(4, width_y));
    OCL_CHECK(err, err = krnl.setArg(5, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (newwidth * newheight * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, cv::COLOR_RGB2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[3], 1);

    absdiff(outputimg0, refimage, error_img0);

#endif

#if NV212YUV4

    inputimg0 = cv::imread(argv[1], 0);
    if (!inputimg0.data) {
        return -1;
    }
    inputimg1 = cv::imread(argv[2], -1);
    if (!inputimg1.data) {
        return -1;
    }

    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;

    int newwidth_u_v = inputimg1.cols << 1;
    int newheight_u_v = inputimg1.rows << 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height_y = inputimg0.rows;
    int width_y = inputimg0.cols;
    int height_u_y = inputimg1.rows;
    int width_u_y = inputimg1.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_nv212yuv4", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicey(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageToDeviceuv(context, CL_MEM_READ_ONLY, (inputimg1.rows * inputimg1.cols * 2), NULL, &err));

    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(
        err, err = q.enqueueWriteBuffer(imageToDevicey, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols), inputimg0.data));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuv, CL_TRUE, 0, (inputimg1.rows * inputimg1.cols * 2),
                                              inputimg1.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicey));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(4, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(5, height_y));
    OCL_CHECK(err, err = krnl.setArg(6, width_y));
    OCL_CHECK(err, err = krnl.setArg(7, height_u_y));
    OCL_CHECK(err, err = krnl.setArg(8, width_u_y));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[3], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[4], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[5], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif

#if RGBA2YUV4

    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGBA);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgba2yuv4", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergba(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 4), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergba, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 4),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergba));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif

#if RGBA2IYUV

    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows >> 2;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGBA);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgba2iyuv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergba(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 4), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergba, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 4),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergba));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

    imwrite("out_Y_error.png", error_img0);
    imwrite("out_U_error.png", error_img1);
    imwrite("out_V_error.png", error_img2);

#endif

#if RGBA2NV12

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    cvtColor(inputimg0, inputimg0, cv::COLOR_BGR2RGBA);
    img_height = inputimg0.rows;
    img_width = inputimg0.cols;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgba2nv12", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergba(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols * 4), NULL,
                                                &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergba, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols * 4),
                                              inputimg0.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergba));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif

#if RGBA2NV21

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    cvtColor(inputimg0, inputimg0, cv::COLOR_BGR2RGBA);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgba2nv21", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergba(context, CL_MEM_READ_ONLY, (inputimg0.rows * inputimg0.cols * 4), NULL,
                                                &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergba, CL_TRUE, 0, (inputimg0.rows * inputimg0.cols * 4),
                                              inputimg0.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergba));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_VU.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif

#if RGB2IYUV

    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows >> 2;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGB);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2iyuv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

    imwrite("out_Y_error.png", error_img0);
    imwrite("out_U_error.png", error_img1);
    imwrite("out_V_error.png", error_img2);

#endif
#if RGB2NV12

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    cvtColor(inputimg0, inputimg0, cv::COLOR_BGR2RGB);
    img_height = inputimg0.rows;
    img_width = inputimg0.cols;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2nv12", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), inputimg0.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_UV.png", outputimg1);
    cv::imwrite("out_Y.png", outputimg0);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif
#if BGR2NV12

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);
    img_height = inputimg0.rows;
    img_width = inputimg0.cols;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2nv12", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), inputimg0.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_UV.png", outputimg1);
    cv::imwrite("out_Y.png", outputimg0);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif
#if RGB2NV21

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    cvtColor(inputimg0, inputimg0, cv::COLOR_BGR2RGB);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2nv21", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), inputimg0.data));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_VU.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif

#if BGR2NV21

    inputimg0 = cv::imread(argv[1], 1);
    if (!inputimg0.data) {
        return -1;
    }
    int newwidth_y = inputimg0.cols;
    int newheight_y = inputimg0.rows;
    int newwidth_uv = inputimg0.cols >> 1;
    int newheight_uv = inputimg0.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_uv, newheight_uv);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg0.rows;
    int width = inputimg0.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2nv21", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newwidth_uv * newheight_uv * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg0.rows * inputimg0.cols * INPUT_CH_TYPE), inputimg0.data));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_uv));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_uv));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_uv * newheight_uv * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_VU.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

#endif

#if RGB2YUV4

    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGB);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2yuv4", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newwidth_u_v * newheight_u_v), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif

#if RGB2YUYV

    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGB);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2yuyv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 3), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y * 2), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 3),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth_y * newheight_y * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_YUYV.png", outputimg0);

    refimage0 = cv::imread(argv[2], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open YUYV reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
#endif

#if BGR2YUYV

    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2yuyv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 3), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y * 2), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 3),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth_y * newheight_y * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_YUYV.png", outputimg0);

    refimage0 = cv::imread(argv[2], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open YUYV reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
#endif

#if RGB2UYVY

    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGB);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2uyvy", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 3), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y * 2), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 3),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth_y * newheight_y * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_YUYV.png", outputimg0);
    refimage0 = cv::imread(argv[2], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open YUYV reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
#endif

#if BGR2UYVY
    inputimg = cv::imread(argv[1], 1);

    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2uyvy", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 3), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y * 2), NULL, &err));

    printf("finished buffer creation task\n");
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 3),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (newwidth_y * newheight_y * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cv::imwrite("out_YUYV.png", outputimg0);
    refimage0 = cv::imread(argv[2], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open YUYV reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
#endif

#if UYVY2IYUV
    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows >> 2;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);
    error_img0.create(S0, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2iyuv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));
    OCL_CHECK(err, err = krnl.setArg(10, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(11, newwidth_u_v));
    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_U.png", outputimg1);
    cv::imwrite("out_V.png", outputimg2);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    imwrite("out_Y_error.png", error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    imwrite("out_U_error.png", error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);
    imwrite("out_V_error.png", error_img2);
#endif

#if UYVY2NV12

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols >> 1;
    int newheight_u_v = inputimg.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2nv12", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newheight_u_v * newwidth_u_v * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);
#endif
#if UYVY2NV21

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols >> 1;
    int newheight_u_v = inputimg.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2nv21", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v * 2), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newheight_u_v * newwidth_u_v * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);
#endif
#if UYVY2YUYV

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_16UC1);
    error_img0.create(S0, CV_16UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2yuyv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceyuyv(context, CL_MEM_WRITE_ONLY, (inputimg.rows * inputimg.cols * 2), NULL,
                                                  &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_YUYV.png", outputimg0);

    refimage0 = cv::imread(argv[2], -1);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open YUYV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
#endif
#if UYVY2RGBA

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }

    int newwidth = inputimg.cols;
    int newheight = inputimg.rows;
    cv::Mat outputimgrgba;
    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC4);
    error_img0.create(S0, CV_8UC4);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2rgba", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergba(context, CL_MEM_WRITE_ONLY, (inputimg.rows * inputimg.cols * 4), NULL,
                                                  &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergba));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergba, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 4),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, cv::COLOR_RGBA2BGR);
    refimage = cv::imread(argv[2], 1);
    if (!refimage.data) {
        fprintf(stderr, "Failed to open reference image\n ");
        return -1;
    }
    absdiff(outputimg0, refimage, error_img0);
#endif

#if UYVY2RGB

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }

    int newwidth = inputimg.cols;
    int newheight = inputimg.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_uyvy2rgb", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceuyvy(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY, (inputimg.rows * inputimg.cols * 3), NULL,
                                                 &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 3),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, cv::COLOR_RGB2BGR);

    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[2], 1);
    if (!refimage.data) {
        fprintf(stderr, "Failed to open reference image\n ");
        return -1;
    }
    absdiff(outputimg0, refimage, error_img0);

#endif

#if YUYV2IYUV
    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols;
    int newheight_u_v = inputimg.rows >> 2;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_8UC1);
    outputimg2.create(S1, CV_8UC1);

    error_img0.create(S0, CV_8UC1);
    error_img1.create(S1, CV_8UC1);
    error_img2.create(S1, CV_8UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL
    ////////////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2iyuv", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newwidth_y * newheight_y), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceu(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicev(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceu));
    OCL_CHECK(err, err = krnl.setArg(3, imageFromDevicev));
    OCL_CHECK(err, err = krnl.setArg(4, height));
    OCL_CHECK(err, err = krnl.setArg(5, width));
    OCL_CHECK(err, err = krnl.setArg(6, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(7, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(8, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(9, newwidth_u_v));
    OCL_CHECK(err, err = krnl.setArg(10, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(11, newwidth_u_v));
    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceu, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicev, CL_TRUE, 0, (newwidth_u_v * newheight_u_v),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg2.data));

    q.finish();
    printf("write output buffer\n");

    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Failed to open Y reference image\n ");
        return (1);
    }
    refimage1 = cv::imread(argv[3], 0);
    if (!refimage1.data) {
        fprintf(stderr, "Failed to open U reference image\n ");
        return (1);
    }
    refimage2 = cv::imread(argv[4], 0);
    if (!refimage2.data) {
        fprintf(stderr, "Failed to open V reference image\n ");
        return (1);
    }

    cv::absdiff(refimage0, outputimg0, error_img0);
    cv::absdiff(refimage1, outputimg1, error_img1);
    cv::absdiff(refimage2, outputimg2, error_img2);

#endif

#if YUYV2NV12

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols; //>>1;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols >> 1;
    int newheight_u_v = inputimg.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2nv12", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_u_v * newheight_u_v * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    printf("\n Written output images\n");
    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

    imwrite("error_Y.png", error_img0);
    imwrite("error_UV.png", error_img1);
#endif
#if YUYV2NV21

    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth_y = inputimg.cols; //>>1;
    int newheight_y = inputimg.rows;
    int newwidth_u_v = inputimg.cols >> 1;
    int newheight_u_v = inputimg.rows >> 1;

    cv::Size S0(newwidth_y, newheight_y);
    outputimg0.create(S0, CV_8UC1);
    error_img0.create(S0, CV_8UC1);

    cv::Size S1(newwidth_u_v, newheight_u_v);
    outputimg1.create(S1, CV_16UC1);
    error_img1.create(S1, CV_16UC1);

    img_width = inputimg.cols;
    img_height = inputimg.rows;

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2nv21", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicey(context, CL_MEM_WRITE_ONLY, (newheight_y * newwidth_y), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDeviceuv(context, CL_MEM_WRITE_ONLY, (newheight_u_v * newwidth_u_v * 2), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicey));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDeviceuv));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));
    OCL_CHECK(err, err = krnl.setArg(5, newheight_y));
    OCL_CHECK(err, err = krnl.setArg(6, newwidth_y));
    OCL_CHECK(err, err = krnl.setArg(7, newheight_u_v));
    OCL_CHECK(err, err = krnl.setArg(8, newwidth_u_v));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicey, CL_TRUE, 0, (newwidth_y * newheight_y),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuv, CL_TRUE, 0, (newwidth_u_v * newheight_u_v * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg1.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cv::imwrite("out_Y.png", outputimg0);
    cv::imwrite("out_UV.png", outputimg1);

    printf("\n Written output images\n");
    refimage0 = cv::imread(argv[2], 0);
    if (!refimage0.data) {
        fprintf(stderr, "Can't open Y ref image !!\n ");
        return -1;
    }

    refimage1 = cv::imread(argv[3], -1);
    if (!refimage1.data) {
        fprintf(stderr, "Can't open UV ref image !!\n ");
        return -1;
    }

    absdiff(outputimg0, refimage0, error_img0);
    absdiff(outputimg1, refimage1, error_img1);

    imwrite("error_Y.png", error_img0);
    imwrite("error_UV.png", error_img1);
#endif

#if YUYV2RGBA
    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth = inputimg.cols;
    int newheight = inputimg.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC4);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2rgba", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergba(context, CL_MEM_WRITE_ONLY, (inputimg.rows * inputimg.cols * 4), NULL,
                                                  &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergba));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergba, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 4),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    cvtColor(outputimg0, outputimg0, cv::COLOR_RGBA2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[2], 1);
    if (!refimage.data) {
        fprintf(stderr, "Failed to read reference image\n ");
        return -1;
    }
    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);
#endif
#if YUYV2UYVY
    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth = inputimg.cols;
    int newheight = inputimg.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_16UC1);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2uyvy", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceuyvy(context, CL_MEM_WRITE_ONLY, (inputimg.rows * inputimg.cols * 2), NULL,
                                                  &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceuyvy));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceuyvy, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    imwrite("out.png", outputimg0);
    refimage = cv::imread(argv[2], -1);
    if (!refimage.data) {
        fprintf(stderr, "Failed to read reference image\n ");
        return -1;
    }
    absdiff(outputimg0, refimage, error_img0);
    imwrite("error_img0.png", error_img0);
#endif
#if YUYV2RGB
    inputimg = cv::imread(argv[1], -1);
    if (!inputimg.data) {
        return -1;
    }
    int newwidth = inputimg.cols;
    int newheight = inputimg.rows;

    cv::Size S0(newwidth, newheight);
    outputimg0.create(S0, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_yuyv2rgb", &err));

    OCL_CHECK(err,
              cl::Buffer imageToDeviceyuyv(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (inputimg.rows * inputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceyuyv, CL_TRUE, 0, (inputimg.rows * inputimg.cols * 2),
                                              inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceyuyv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(
        err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0, (inputimg.rows * inputimg.cols * OUTPUT_CH_TYPE),
                                       (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg0.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    cvtColor(outputimg0, outputimg0, cv::COLOR_RGB2BGR);
    imwrite("out.png", outputimg0);

    refimage = cv::imread(argv[2], 1);
    if (!refimage.data) {
        fprintf(stderr, "Failed to read reference image\n ");
        return -1;
    }
    absdiff(outputimg0, refimage, error_img0);

#endif
#if RGB2GRAY
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC1);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC1);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2gray", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicegray(context, CL_MEM_WRITE_ONLY,
                                                  (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */

    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicegray));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicegray, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2GRAY, 1);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if BGR2GRAY
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC1);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC1);
    /////////////////////////////////////// CL
    /////////////////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2gray", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicebgr(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicegray(context, CL_MEM_WRITE_ONLY,
                                                  (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicebgr, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicegray));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicegray, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL
    ////////////////////////////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_BGR2GRAY, 1);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if GRAY2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 0);
    if (!inputimg.data) {
        return -1;
    }
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_gray2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicegray(context, CL_MEM_READ_ONLY,
                                                (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicegray, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicegray));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_GRAY2RGB, 3);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    cv::absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if GRAY2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 0);
    if (!inputimg.data) {
        return -1;
    }
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_gray2bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicegray(context, CL_MEM_READ_ONLY,
                                                (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicebgr(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicegray, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicegray));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicebgr, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_GRAY2BGR, 3);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if RGB2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicexyz(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicexyz, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::imwrite("ocv_out.jpg", inputimg1);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, inputimg1, error_img0);

#endif
#if BGR2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicexyz(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicexyz, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, inputimg, cv::COLOR_BGR2RGB);

    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, inputimg, error_img0);

#endif
#if RGB2XYZ
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2xyz", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicexyz(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicexyz, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2XYZ);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if BGR2XYZ
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL
    ////////////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2xyz", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicebgr(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicexyz(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicebgr, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicexyz, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_BGR2XYZ);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if XYZ2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);
    cv::cvtColor(inputimg, inputimg, cv::COLOR_RGB2XYZ);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_xyz2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicexyz(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicexyz, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2XYZ);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_XYZ2RGB);

    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if XYZ2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2XYZ);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_xyz2bgr", &err));
    OCL_CHECK(err, cl::Buffer imageToDevicexyz(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicebgr(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicexyz, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicexyz));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicebgr, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_XYZ2BGR);

    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if RGB2YCrCb
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2ycrcb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceycrcb(context, CL_MEM_WRITE_ONLY,
                                                   (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");
    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceycrcb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceycrcb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2YCrCb);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if BGR2YCrCb
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2ycrcb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicebgr(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceycrcb(context, CL_MEM_WRITE_ONLY,
                                                   (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicebgr, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDeviceycrcb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDeviceycrcb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));

    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_BGR2YCrCb);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if YCrCb2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2YCrCb);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_ycrcb2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDeviceycrcb(context, CL_MEM_READ_ONLY,
                                                 (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceycrcb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceycrcb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_YCrCb2RGB);

    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if YCrCb2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2YCrCb);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_ycrcb2bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDeviceycrcb(context, CL_MEM_READ_ONLY,
                                                 (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicebgr(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDeviceycrcb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceycrcb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicebgr, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_YCrCb2BGR);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);
    cv::imwrite("error_img0.jpg", error_img0);
#endif
#if RGB2HLS
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2hls", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicehls(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicehls));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicehls, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2HLS);

    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if BGR2HLS
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2hls", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicebgr(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicehls(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicebgr, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicehls));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicehls, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_BGR2HLS);
    // c_Ref((float*)inputimg.data,(float*)ocv_outputimg.data,3);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if HLS2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    cv::cvtColor(inputimg, inputimg, cv::COLOR_BGR2HLS);
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_hls2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicehls(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicehls, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicehls));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////
    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_HLS2RGB);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if HLS2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    cv::cvtColor(inputimg, inputimg, cv::COLOR_BGR2HLS);
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    /////////////////////////////////////// CL ////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_hls2bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicehls(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicebgr(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicehls, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicehls));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicebgr, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL ////////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_HLS2BGR);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if RGB2HSV
    cv::Mat outputimg, ocv_outputimg;
    inputimg1 = cv::imread(argv[1], 1);
    if (!inputimg1.data) {
        return -1;
    }
    cv::cvtColor(inputimg1, inputimg, cv::COLOR_BGR2RGB);

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    static xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, XF_NPPC1> imgInput(inputimg.rows, inputimg.cols);
    static xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, XF_NPPC1> imgOutput(outputimg.rows, outputimg.cols);

    //////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_rgb2hsv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicergb(context, CL_MEM_READ_ONLY, (inputimg.rows * inputimg.cols), NULL, &err));
    OCL_CHECK(err,
              cl::Buffer imageFromDevicehsv(context, CL_MEM_WRITE_ONLY, (outputimg.rows * outputimg.cols), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicergb, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));

    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicergb));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicehsv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicehsv, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_RGB2HSV);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if BGR2HSV
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }

    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    //////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_bgr2hsv", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicebgr(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicehsv(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicebgr, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicehsv));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicehsv, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_BGR2HSV);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if HSV2RGB
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    cv::cvtColor(inputimg, inputimg, cv::COLOR_BGR2HSV);
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    //////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_hsv2rgb", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicehsv(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicergb(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicehsv, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicehsv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicergb));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicergb, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_HSV2RGB);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
#if HSV2BGR
    cv::Mat outputimg, ocv_outputimg;
    inputimg = cv::imread(argv[1], 1);
    if (!inputimg.data) {
        return -1;
    }
    cv::cvtColor(inputimg, inputimg, cv::COLOR_BGR2HSV);
    outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);
    ocv_outputimg.create(inputimg.rows, inputimg.cols, CV_8UC3);

    //////////////////////////////////////////// CL
    //////////////////////////////////////
    int height = inputimg.rows;
    int width = inputimg.cols;

    OCL_CHECK(err, cl::Kernel krnl(program, "cvtcolor_hsv2bgr", &err));

    OCL_CHECK(err, cl::Buffer imageToDevicehsv(context, CL_MEM_READ_ONLY,
                                               (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevicebgr(context, CL_MEM_WRITE_ONLY,
                                                 (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    /* Copy input vectors to memory */
    OCL_CHECK(err, err = q.enqueueWriteBuffer(imageToDevicehsv, CL_TRUE, 0,
                                              (inputimg.rows * inputimg.cols * INPUT_CH_TYPE), inputimg.data));
    printf("finished enqueueing task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevicehsv));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevicebgr));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    printf("finished setting kernel arguments\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");
    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
    OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevicebgr, CL_TRUE, 0,
                                             (outputimg.rows * outputimg.cols * OUTPUT_CH_TYPE),
                                             (ap_uint<OUTPUT_PTR_WIDTH>*)outputimg.data));
    q.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////

    // OpenCV reference
    cv::cvtColor(inputimg, ocv_outputimg, cv::COLOR_HSV2BGR);
    cv::imwrite("ocv_out.jpg", ocv_outputimg);
    cv::imwrite("hls_out.jpg", outputimg);
    absdiff(outputimg, ocv_outputimg, error_img0);

#endif
    double minval, maxval;
    float err_per;
    int cnt;

    minval = 255;
    maxval = 0;
    cnt = 0;
    for (int i = 0; i < error_img0.rows; i++) {
        for (int j = 0; j < error_img0.cols; j++) {
            uchar v = error_img0.at<uchar>(i, j);

            if (v > ERROR_THRESHOLD) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    err_per = 100.0 * (float)cnt / (error_img0.rows * error_img0.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 3.0f) {
        fprintf(stderr, "\n1st Image Test Failed\n ");
        return 1;
    }

#if (IYUV2NV12 || RGBA2NV12 || RGB2NV12 || BGR2NV12 || BGR2NV21 || RGB2NV21 || RGBA2NV21 || UYVY2NV12 || YUYV2NV12 || \
     NV122IYUV || NV212IYUV || IYUV2YUV4 || NV122YUV4 || NV212YUV4 || RGBA2IYUV || RGBA2YUV4 || UYVY2IYUV ||          \
     YUYV2IYUV || RGB2IYUV || RGB2YUV4 || NV122NV21 || NV212NV12)
    minval = 255;
    maxval = 0;
    cnt = 0;
    for (int i = 0; i < error_img1.rows; i++) {
        for (int j = 0; j < error_img1.cols; j++) {
            uchar v = error_img1.at<uchar>(i, j);

            if (v > ERROR_THRESHOLD) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    err_per = 100.0 * (float)cnt / (error_img1.rows * error_img1.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;
    if (err_per > 3.0f) {
        fprintf(stderr, "\n2nd Image Test Failed\n ");
        return 1;
    }

#endif
#if (IYUV2YUV4 || NV122IYUV || NV122YUV4 || NV212IYUV || NV212YUV4 || RGBA2IYUV || RGB2IYUV || RGBA2YUV4 || \
     UYVY2IYUV || YUYV2IYUV || RGB2YUV4)
    minval = 255;
    maxval = 0;
    cnt = 0;
    for (int i = 0; i < error_img2.rows; i++) {
        for (int j = 0; j < error_img2.cols; j++) {
            uchar v = error_img2.at<uchar>(i, j);

            if (v > ERROR_THRESHOLD) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    err_per = 100.0 * (float)cnt / (error_img2.rows * error_img2.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;
    if (err_per > 3.0f) {
        fprintf(stderr, "\n3rd Image Test Failed\n ");
        return 1;
    }
#endif
    /* ## *************************************************************** ##*/
    return 0;
}
