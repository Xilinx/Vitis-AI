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
#include "xf_hist_equalize_config.h"

#include "xcl2.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_img_copy, out_img, ocv_ref, diff;

    // reading in the color image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }

    int height = in_img.rows;
    int width = in_img.cols;

    // create memory for output images
    in_img.copyTo(in_img_copy);
    out_img.create(height, width, XF_8UC1);
    ocv_ref.create(height, width, XF_8UC1);
    diff.create(height, width, XF_8UC1);

    ///////////////// 	Opencv  Reference  ////////////////////////
    cv::equalizeHist(in_img, ocv_ref);

    /////////////////////////////////////// CL ////////////////////////

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_hist_equalize");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "equalizeHist_accel");

    cl::Buffer imageToDevice1(context, CL_MEM_READ_ONLY, height * width);
    cl::Buffer imageToDevice2(context, CL_MEM_READ_ONLY, height * width);
    cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, height * width);

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice1);
    krnl.setArg(1, imageToDevice2);
    krnl.setArg(2, imageFromDevice);
    krnl.setArg(3, height);
    krnl.setArg(4, width);

    q.enqueueWriteBuffer(imageToDevice1, CL_TRUE, 0, height * width, in_img.data);
    q.enqueueWriteBuffer(imageToDevice2, CL_TRUE, 0, height * width, in_img_copy.data);

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, height * width, out_img.data);
    q.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    //////////////////  Compute Absolute Difference ////////////////////
    cv::absdiff(ocv_ref, out_img, diff);

    cv::imwrite("input.jpg", in_img);
    cv::imwrite("out_ocv.jpg", ocv_ref);
    cv::imwrite("out_hls.jpg", out_img);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 0.0f) {
        return 1;
    }
    return 0;
}
