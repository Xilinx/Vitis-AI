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
#include "xf_remap_config.h"

#define READ_MAPS_FROM_FILE 0

int main(int argc, char** argv) {
#if READ_MAPS_FROM_FILE
    if (argc != 4) {
        fprintf(stderr, "Usage: <executable> <input image path> <mapx file> <mapy file>\n");
        return -1;
    }
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image path>\n");
        return -1;
    }
#endif

    cv::Mat src, ocv_remapped, hls_remapped;
    cv::Mat map_x, map_y, diff;

// Reading in the image:
#if GRAY
    src = cv::imread(argv[1], 0); // read image Grayscale
#else
    src = cv::imread(argv[1], 1); // read image RGB
#endif

    if (!src.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Allocate memory for the outputs:
    std::cout << "INFO: Allocate memory for input and output data." << std::endl;
    ocv_remapped.create(src.rows, src.cols, src.type()); // opencv result
    map_x.create(src.rows, src.cols, CV_32FC1);          // Mapx for opencv remap function
    map_y.create(src.rows, src.cols, CV_32FC1);          // Mapy for opencv remap function
    hls_remapped.create(src.rows, src.cols, src.type()); // create memory for output images
    diff.create(src.rows, src.cols, src.type());

// Initialize the float maps:
#if READ_MAPS_FROM_FILE
    // read the float map data from the file (code could be alternated for reading
    // from image)
    FILE *fp_mx, *fp_my;
    fp_mx = fopen(argv[2], "r");
    fp_my = fopen(argv[3], "r");
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float valx, valy;
            if (fscanf(fp_mx, "%f", &valx) != 1) {
                fprintf(stderr, "Not enough data in the provided map_x file ... !!!\n ");
            }
            if (fscanf(fp_my, "%f", &valy) != 1) {
                fprintf(stderr, "Not enough data in the provided map_y file ... !!!\n ");
            }
            map_x.at<float>(i, j) = valx;
            map_y.at<float>(i, j) = valy;
        }
    }
#else // example map generation, flips the image horizontally
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float valx = (float)(src.cols - j - 1), valy = (float)i;
            map_x.at<float>(i, j) = valx;
            map_y.at<float>(i, j) = valy;
        }
    }
#endif

    // Opencv reference:
    std::cout << "INFO: Run reference function in CV." << std::endl;
#if INTERPOLATION == 0
    cv::remap(src, ocv_remapped, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
#else
    cv::remap(src, ocv_remapped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
#endif

    // OpenCL section:
    size_t image_in_size_bytes = src.rows * src.cols * sizeof(unsigned char) * CHANNELS;
    size_t map_in_size_bytes = src.rows * src.cols * sizeof(float);
    size_t image_out_size_bytes = image_in_size_bytes;

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_remap");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "remap_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inMapX(context, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inMapY(context, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    int rows = src.rows;
    int cols = src.cols;
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inMapX));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inMapY));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(4, rows));
    OCL_CHECK(err, err = kernel.setArg(5, cols));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       src.data,            // Pointer to the data to copy
                                       nullptr, &event));

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inMapX,     // buffer on the FPGA
                                       CL_TRUE,           // blocking call
                                       0,                 // buffer offset in bytes
                                       map_in_size_bytes, // Size in bytes
                                       map_x.data,        // Pointer to the data to copy
                                       nullptr, &event));

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inMapY,     // buffer on the FPGA
                                       CL_TRUE,           // blocking call
                                       0,                 // buffer offset in bytes
                                       map_in_size_bytes, // Size in bytes
                                       map_y.data,        // Pointer to the data to copy
                                       nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            hls_remapped.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Save the results:
    cv::imwrite("ocv_reference_out.jpg", ocv_remapped); // Opencv Result
    cv::imwrite("kernel_out.jpg", hls_remapped);

    // Results verification:
    cv::absdiff(ocv_remapped, hls_remapped, diff);
    cv::imwrite("diff.png", diff);

    // Find minimum and maximum differences.
    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
