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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_gray, in_gray1, out_gray, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    int channels = in_gray.channels();
    int height = in_gray.rows;
    int width = in_gray.cols;
    // OpenCV function
    std::vector<double> ocv_scl(channels);

    for (int i = 0; i < channels; ++i) ocv_scl[i] = cv::sum(in_gray)[i];

    // OpenCL section:
    std::vector<double> hls_sum(channels);

    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t vec_out_size_bytes = (channels) * sizeof(double);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_sum");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "sum_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outVec(context, CL_MEM_WRITE_ONLY, vec_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outVec));
    OCL_CHECK(err, err = kernel.setArg(2, height));
    OCL_CHECK(err, err = kernel.setArg(3, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray.data,        // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outVec, // This buffers data will be read
                            CL_TRUE,       // blocking call
                            0,             // offset
                            vec_out_size_bytes,
                            hls_sum.data(), // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Results verification:
    int cnt = 0;

    for (int i = 0; i < in_gray.channels(); i++) {
        if (abs(double(ocv_scl[i]) - hls_sum[i]) > 0.01f) cnt++;
    }

    if (cnt > 0) {
        fprintf(stderr, "INFO: Error percentage = %d%. Test Failed.\n ", 100.0 * float(cnt) / float(channels));
        return EXIT_FAILURE;
    } else
        std::cout << "Test Passed." << std::endl;

    return 0;
}
