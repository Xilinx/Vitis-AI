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
#include "xf_letterbox_config.h"

#include <sys/time.h>
#include "xcl2.hpp"

// letterbox_utils.h files is used for software letterbox implementation
#include "letterbox_utils.h"

int main(int argc, char* argv[]) {
    struct timeval start_pp_sw, end_pp_sw;
    double lat_pp_sw = 0.0f;
    cv::Mat img, result_hls, result_hls_bgr, result_hls_8bit, result_ocv, result_ocv_resize, error;

    img = cv::imread(argv[1], 1);
    if (!img.data) {
        fprintf(stderr, "\n image not found");
        return -1;
    }
    int in_width, in_height;
    int out_width_resize, out_height_resize;
    int out_width, out_height;

    in_width = img.cols;
    in_height = img.rows;

    out_height = 64;
    out_width = 80;
    // Compute Resize output image size for Letterbox
    float scale_height = (float)out_height / (float)in_height;
    float scale_width = (float)out_width / (float)in_width;
    if (scale_width < scale_height) {
        out_width_resize = out_width;
        out_height_resize = (int)((float)(in_height * out_width) / (float)in_width);
    } else {
        out_width_resize = (int)((float)(in_width * out_height) / (float)in_height);
        out_height_resize = out_height;
    }

    /* 128 corresponds to grey pxel */
    int insert_pad_value = 128;

    result_hls.create(cv::Size(out_width, out_height), CV_8UC3);
    result_ocv.create(cv::Size(out_width, out_height), CV_8UC3);

    cl_int err;
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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_letterbox");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "letterbox_accel", &err));

    // Allocate the buffers:
    std::vector<cl::Memory> inBufVec, outBufVec;
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, in_height * in_width * 3, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, out_height * out_width * 3, NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevice));
    OCL_CHECK(err, err = krnl.setArg(2, in_height));
    OCL_CHECK(err, err = krnl.setArg(3, in_width));
    OCL_CHECK(err, err = krnl.setArg(4, out_height));
    OCL_CHECK(err, err = krnl.setArg(5, out_width));
    OCL_CHECK(err, err = krnl.setArg(6, insert_pad_value));

    /* Copy input vectors to memory */
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice,            // buffer on the FPGA
                                        CL_TRUE,                  // blocking call
                                        0,                        // buffer offset in bytes
                                        in_height * in_width * 3, // Size in bytes
                                        img.data));               // Pointer to the data to copy

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Execute the kernel:
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));

    // std::cout << "INFO: Enque task DONE "  << std::endl;
    clWaitForEvents(1, (const cl_event*)&event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    OCL_CHECK(err, q.enqueueReadBuffer(imageFromDevice, // This buffers data will be read
                                       CL_TRUE,         // blocking call
                                       0,               // offset
                                       out_height * out_width * 3,
                                       result_hls.data)); // Data will be stored here

    q.finish();
    /////////////////////////////////////// end of CL ///////////////////////////////////////

    /*Reference Implementation in software*/
    cv::resize(img, result_ocv_resize, cv::Size(out_width_resize, out_height_resize));
    image boxed = make_image(out_width, out_height, 3);
    fill_image(boxed, 0.5);
    image resized = load_image_cv(result_ocv_resize);
    embed_image(resized, boxed, (out_width - out_width_resize) / 2, (out_height - out_height_resize) / 2);
    write_image_cv(boxed, result_ocv);

    /* save output images */
    imwrite("output_ocv.jpg", result_ocv); // Opencv Image
    imwrite("output_hls.jpg", result_hls); // Hls Image

    /* Error check */
    float max_error = -100;
    for (int i = 0; i < (out_width * out_height * 3); i++) {
        float error1 = fabs((float)result_hls.data[i] - (float)result_ocv.data[i]);
        if (error1 > max_error) max_error = error1;
    }

    if (max_error > 2) {
        fprintf(stderr, "\n Test Failed\n");
        return -1;

    } else {
        std::cout << "Test Passed " << std::endl;
        return 0;
    }
}
