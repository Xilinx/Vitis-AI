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
#include "xcl2.hpp"
#include "xf_min_max_loc_config.h"
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_gray, in_conv;

    // Reading in the image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

// Pixel depth conversion:
#if T_8U
    in_img.convertTo(in_conv, CV_8UC1);
#elif T_16U
    in_img.convertTo(in_conv, CV_16UC1);
#elif T_16S
    in_img.convertTo(in_conv, CV_16SC1);
#elif T_32S
    in_img.convertTo(in_conv, CV_32SC1);
#endif

    double cv_minval = 0, cv_maxval = 0;
    cv::Point cv_minloc, cv_maxloc;

    // OpenCV reference:
    cv::minMaxLoc(in_conv, &cv_minval, &cv_maxval, &cv_minloc, &cv_maxloc, cv::noArray());

    // Data for holding outputs from kernel:
    int32_t min_value, max_value;
    std::vector<int32_t> min_max_value(2);

    uint16_t _min_locx, _min_locy, _max_locx, _max_locy;
    std::vector<uint16_t> min_max_loc_xy(4);

    int height = in_img.rows;
    int width = in_img.cols;

    // OpenCL section:
    size_t image_in_size_bytes = height * width * sizeof(INTYPE);
    size_t vec1_out_size_bytes = min_max_value.size() * sizeof(int32_t);
    size_t vec2_out_size_bytes = min_max_loc_xy.size() * sizeof(int32_t);

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_minmaxloc");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "minmaxloc_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outVec1(context, CL_MEM_WRITE_ONLY, vec1_out_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outVec2(context, CL_MEM_WRITE_ONLY, vec2_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outVec1));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outVec2));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_conv.data,        // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outVec1, // This buffers data will be read
                            CL_TRUE,        // blocking call
                            0,              // offset
                            vec1_out_size_bytes,
                            min_max_value.data(), // Data will be stored here
                            nullptr, &event);

    queue.enqueueReadBuffer(buffer_outVec2, // This buffers data will be read
                            CL_TRUE,        // blocking call
                            0,              // offset
                            vec2_out_size_bytes,
                            min_max_loc_xy.data(), // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Split the data from vectors for verification:
    min_value = min_max_value[0];
    max_value = min_max_value[1];
    _min_locx = min_max_loc_xy[0];
    _min_locy = min_max_loc_xy[1];
    _max_locx = min_max_loc_xy[2];
    _max_locy = min_max_loc_xy[3];

    // OpenCV output:
    std::cout << "INFO: Results verification:" << std::endl;
    std::cout << "\tOCV-Minvalue = " << cv_minval << std::endl;
    std::cout << "\tOCV-Maxvalue = " << cv_maxval << std::endl;
    std::cout << "\tOCV-Min Location.x = " << cv_minloc.x << "  OCV-Min Location.y = " << cv_minloc.y << std::endl;
    std::cout << "\tOCV-Max Location.x = " << cv_maxloc.x << "  OCV-Max Location.y = " << cv_maxloc.y << std::endl;

    // Kernel output:
    std::cout << "\tHLS-Minvalue = " << min_value << std::endl;
    std::cout << "\tHLS-Maxvalue = " << max_value << std::endl;
    std::cout << "\tHLS-Min Location.x = " << _min_locx << "  HLS-Min Location.y = " << _min_locy << std::endl;
    std::cout << "\tHLS-Max Location.x = " << _max_locx << "  HLS-Max Location.y = " << _max_locy << std::endl
              << std::endl;

    // Difference in min and max, values and locations of both OpenCV and Kernel
    // function:
    std::cout << "\tDifference in Minimum value: " << (cv_minval - min_value) << std::endl;
    std::cout << "\tDifference in Maximum value: " << (cv_maxval - max_value) << std::endl;
    std::cout << "\tDifference in Minimum value location: (" << (cv_minloc.y - _min_locy) << ","
              << (cv_minloc.x - _min_locx) << ")" << std::endl;
    std::cout << "\tDifference in Maximum value location: (" << (cv_maxloc.y - _max_locy) << ","
              << (cv_maxloc.x - _max_locx) << ")" << std::endl;

    if (((cv_minloc.y - _min_locy) > 1) || ((cv_minloc.x - _min_locx) > 1)) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
