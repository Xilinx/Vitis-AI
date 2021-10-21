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

// Reference implementation:
double GetOtsuThresholdFloat(cv::Mat _src) {
    cv::Size size = _src.size();
    if (_src.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }
    const int N = 256;
    int i, j, h[N] = {0};

    for (i = 0; i < size.height; i++) {
        const unsigned char* src = _src.data + _src.step * i;
        j = 0;
        for (; j < size.width; j++) {
            h[src[j]]++;
        }
    }
    double mu = 0.f;
    double scale;

    scale = 1. / (size.width * size.height);

    for (i = 0; i < N; i += 2) {
        double a = (double)h[i];
        double b = (double)h[i + 1];
        mu += i * (a + b) + b;
    }

    mu = mu * scale;

    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (i = 0; i < N; i++) {
        double p_i, q2, mu2, sigma;

        p_i = h[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON) continue;

        mu1 = (mu1 + i * p_i) / q1;

        mu2 = (mu - q1 * mu1) / q2;

        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}
/***************************************************************************/

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat img, res_img;

    // Reading in the image:
    img = cv::imread(argv[1], 0);

    if (img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Parameters for Otsu:
    double Otsuval_ref;
    uint8_t Otsuval;
    int maxdiff = 0;

    int height = img.rows;
    int width = img.cols;

    res_img = img.clone();

    // Reference function:
    Otsuval_ref = GetOtsuThresholdFloat(res_img);

    // OpenCL section:
    size_t image_in_size_bytes = img.rows * img.cols * sizeof(unsigned char);
    size_t data_out_size_bytes = sizeof(unsigned char);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_otsuthreshold");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "otsuthreshold_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outData(context, CL_MEM_WRITE_ONLY, data_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outData));
    OCL_CHECK(err, err = kernel.setArg(2, height));
    OCL_CHECK(err, err = kernel.setArg(3, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, err = queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                                  CL_TRUE,             // blocking call
                                                  0,                   // buffer offset in bytes
                                                  image_in_size_bytes, // Size in bytes
                                                  img.data,            // Pointer to the data to copy
                                                  nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outData, // This buffers data will be read
                            CL_TRUE,        // blocking call
                            0,              // offset
                            data_out_size_bytes,
                            &Otsuval, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Results verification:
    if (abs(Otsuval_ref - Otsuval) > maxdiff) maxdiff = abs(Otsuval_ref - Otsuval);

    std::cout << "INFO: Otsu threshold results obtained:" << std::endl;
    std::cout << "\tReference: " << (int)Otsuval_ref << "\tHLS Threshold : " << (int)Otsuval
              << "\tDifference : " << (int)maxdiff << std::endl;

    if (maxdiff > 1) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
