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
#include "xf_stereo_pipeline_config.h"
#include "cameraParameters.h"

#include "xcl2.hpp"

#define _PROFILE_ 0

using namespace std;

int main(int argc, char** argv) {
    cv::setUseOptimized(false);

    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage: <executable> <left image> <right image>\n");
        return -1;
    }

    cv::Mat left_img, right_img;
    left_img = cv::imread(argv[1], 0);
    if (left_img.data == NULL) {
        fprintf(stderr, "Cannot open left image at %s\n", argv[1]);
        return 0;
    }
    right_img = cv::imread(argv[2], 0);
    if (right_img.data == NULL) {
        fprintf(stderr, "Cannot open right image at %s\n", argv[1]);
        return 0;
    }

    //////////////////	HLS TOP Function Call  ////////////////////////
    int rows = left_img.rows;
    int cols = left_img.cols;
    cv::Mat disp_img(rows, cols, CV_16UC1);

    // allocate mem for camera parameters for rectification and bm_state class
    float* cameraMA_l_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* cameraMA_r_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* irA_l_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* irA_r_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* distC_l_fl = (float*)malloc(XF_DIST_COEFF_SIZE * sizeof(float));
    float* distC_r_fl = (float*)malloc(XF_DIST_COEFF_SIZE * sizeof(float));
    int* bm_state_arr = (int*)malloc(11 * sizeof(int));

    xf::cv::xFSBMState<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS> bm_state;
    bm_state.uniquenessRatio = 15;
    bm_state.textureThreshold = 20;
    bm_state.minDisparity = 0;
    bm_state_arr[0] = bm_state.preFilterType;
    bm_state_arr[1] = bm_state.preFilterSize;
    bm_state_arr[2] = bm_state.preFilterCap;
    bm_state_arr[3] = bm_state.SADWindowSize;
    bm_state_arr[4] = bm_state.minDisparity;
    bm_state_arr[5] = bm_state.numberOfDisparities;
    bm_state_arr[6] = bm_state.textureThreshold;
    bm_state_arr[7] = bm_state.uniquenessRatio;
    bm_state_arr[8] = bm_state.ndisp_unit;
    bm_state_arr[9] = bm_state.sweepFactor;
    bm_state_arr[10] = bm_state.remainder;

    // copy camera params
    for (int i = 0; i < XF_CAMERA_MATRIX_SIZE; i++) {
        cameraMA_l_fl[i] = (float)cameraMA_l[i];
        cameraMA_r_fl[i] = (float)cameraMA_r[i];
        irA_l_fl[i] = (float)irA_l[i];
        irA_r_fl[i] = (float)irA_r[i];
    }

    // copy distortion coefficients
    for (int i = 0; i < XF_DIST_COEFF_SIZE; i++) {
        distC_l_fl[i] = (float)distC_l[i];
        distC_r_fl[i] = (float)distC_r[i];
    }

    /////////////////////////////////////// CL ////////////////////////
    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Context, command queue and device name:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_stereopipeline");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "stereopipeline_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer imageToDeviceL(context, CL_MEM_READ_ONLY, rows * cols, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDeviceR(context, CL_MEM_READ_ONLY, rows * cols, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, rows * cols * 2, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceCML(context, CL_MEM_READ_ONLY, sizeof(float) * XF_CAMERA_MATRIX_SIZE, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceCMR(context, CL_MEM_READ_ONLY, sizeof(float) * XF_CAMERA_MATRIX_SIZE, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceDCL(context, CL_MEM_READ_ONLY, sizeof(float) * XF_DIST_COEFF_SIZE, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceDCR(context, CL_MEM_READ_ONLY, sizeof(float) * XF_DIST_COEFF_SIZE, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceRAL(context, CL_MEM_READ_ONLY, sizeof(float) * XF_CAMERA_MATRIX_SIZE, NULL, &err));
    OCL_CHECK(err,
              cl::Buffer arrToDeviceRAR(context, CL_MEM_READ_ONLY, sizeof(float) * XF_CAMERA_MATRIX_SIZE, NULL, &err));
    OCL_CHECK(err, cl::Buffer structToDevicesbmstate(context, CL_MEM_READ_ONLY, sizeof(int) * 11, NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDeviceL));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDeviceR));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevice));
    OCL_CHECK(err, err = krnl.setArg(3, arrToDeviceCML));
    OCL_CHECK(err, err = krnl.setArg(4, arrToDeviceCMR));
    OCL_CHECK(err, err = krnl.setArg(5, arrToDeviceDCL));
    OCL_CHECK(err, err = krnl.setArg(6, arrToDeviceDCR));
    OCL_CHECK(err, err = krnl.setArg(7, arrToDeviceRAL));
    OCL_CHECK(err, err = krnl.setArg(8, arrToDeviceRAR));
    OCL_CHECK(err, err = krnl.setArg(9, structToDevicesbmstate));
    OCL_CHECK(err, err = krnl.setArg(10, rows));
    OCL_CHECK(err, err = krnl.setArg(11, cols));

    // Copying input data to Device buffer from host memory
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDeviceL, CL_TRUE, 0, rows * cols, left_img.data));
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDeviceR, CL_TRUE, 0, rows * cols, right_img.data));
    OCL_CHECK(err,
              q.enqueueWriteBuffer(arrToDeviceCML, CL_TRUE, 0, sizeof(float) * XF_CAMERA_MATRIX_SIZE, cameraMA_l_fl));
    OCL_CHECK(err,
              q.enqueueWriteBuffer(arrToDeviceCMR, CL_TRUE, 0, sizeof(float) * XF_CAMERA_MATRIX_SIZE, cameraMA_r_fl));
    OCL_CHECK(err, q.enqueueWriteBuffer(arrToDeviceDCL, CL_TRUE, 0, sizeof(float) * XF_DIST_COEFF_SIZE, distC_l_fl));
    OCL_CHECK(err, q.enqueueWriteBuffer(arrToDeviceDCR, CL_TRUE, 0, sizeof(float) * XF_DIST_COEFF_SIZE, distC_r_fl));
    OCL_CHECK(err, q.enqueueWriteBuffer(arrToDeviceRAL, CL_TRUE, 0, sizeof(float) * XF_CAMERA_MATRIX_SIZE, irA_l_fl));
    OCL_CHECK(err, q.enqueueWriteBuffer(arrToDeviceRAR, CL_TRUE, 0, sizeof(float) * XF_CAMERA_MATRIX_SIZE, irA_r_fl));
    OCL_CHECK(err, q.enqueueWriteBuffer(structToDevicesbmstate, CL_TRUE, 0, sizeof(int) * 11, bm_state_arr));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));

    // profiling
    clWaitForEvents(1, (const cl_event*)&event_sp);
#if _PROFILE_
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;
#endif

    // Copying Device result data to Host memory
    OCL_CHECK(err, q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, rows * cols * 2, disp_img.data));

    q.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    // Write output image
    cv::Mat out_disp_img(rows, cols, CV_8UC1);
    disp_img.convertTo(out_disp_img, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("hls_output.png", out_disp_img);
    printf("run complete !\n");

    return 0;
}
