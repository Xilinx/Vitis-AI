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
#include "xf_svm_config.h"

/*****************************************************************************
 * 	 main function: SVM core
 *****************************************************************************/
int main(int argc, char** argv) {
    float in_1[IN_ARRAY_SIZE_1], in_2[IN_ARRAY_SIZE_2];

    float a = 0, bias = 0.567;

    // Init input data:
    for (int i = 0; i < TOTAL_ARRAY_ELEMENTS; i++) {
        float temp1 = a;
        float temp2 = 0.2 - a;

        if (temp1 >= 1) temp1 = 0.99;
        if (temp2 >= 1) temp2 = 0.99;

        in_1[i] = temp1 * pow(2, IN_FRAC_BITS_1);
        in_2[i] = temp2 * pow(2, IN_FRAC_BITS_2);

        a += 0.0008;
    }

    // Allocate data for the kernel outputs:
    int64_t out_result = 0;
    unsigned char out_frac = 0;

    // OpenCL section:
    std::vector<uint16_t> svm_params = {INDEX_ARR_1, INDEX_ARR_2, IN_FRAC_BITS_1, IN_FRAC_BITS_2,
                                        NO_OF_KERNEL_ELEMENTS};

    size_t input_1_size_bytes = IN_ARRAY_SIZE_1 * sizeof(float);
    size_t input_2_size_bytes = IN_ARRAY_SIZE_2 * sizeof(float);
    size_t vec_in_size_bytes = svm_params.size() * sizeof(uint16_t);
    size_t out_frac_size_bytes = sizeof(unsigned char);
    size_t out_result_size_bytes = sizeof(int64_t);

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Contex, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_svm");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "svm_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage1(context, CL_MEM_READ_ONLY, input_1_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage2(context, CL_MEM_READ_ONLY, input_2_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inVecParams(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outFractional(context, CL_MEM_WRITE_ONLY, out_frac_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outResult(context, CL_MEM_WRITE_ONLY, out_result_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inVecParams));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outFractional));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_outResult));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage1,    // buffer on the FPGA
                                            CL_TRUE,            // blocking call
                                            0,                  // buffer offset in bytes
                                            input_1_size_bytes, // Size in bytes
                                            in_1,               // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage2,    // buffer on the FPGA
                                            CL_TRUE,            // blocking call
                                            0,                  // buffer offset in bytes
                                            input_2_size_bytes, // Size in bytes
                                            in_2,               // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inVecParams, // buffer on the FPGA
                                            CL_TRUE,            // blocking call
                                            0,                  // buffer offset in bytes
                                            vec_in_size_bytes,  // Size in bytes
                                            svm_params.data(),  // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outFractional, // This buffers data will be read
                            CL_TRUE,              // blocking call
                            0,                    // offset
                            out_frac_size_bytes,
                            &out_frac, // Data will be stored here
                            nullptr, &event);

    queue.enqueueReadBuffer(buffer_outResult, // This buffers data will be read
                            CL_TRUE,          // blocking call
                            0,                // offset
                            out_result_size_bytes,
                            &out_result, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Calculate the bias fix for the result:
    int bias_fix = bias * pow(2, out_frac);
    long int resultFIX_bias = out_result + bias_fix;

    // Calculate the reference result:
    float float_res_fix = (double)resultFIX_bias / pow(2, out_frac);

    // OpenCV reference function
    cv::Mat A(cv::Size(1, IN_ARRAY_SIZE_1), CV_16S);
    std::memcpy(A.data, in_1, IN_ARRAY_SIZE_1 * sizeof(ap_int<16>));
    cv::Mat B(cv::Size(1, IN_ARRAY_SIZE_1), CV_16S);
    std::memcpy(B.data, in_2, IN_ARRAY_SIZE_1 * sizeof(ap_int<16>));
    long int ocv_fix = (A.dot(B));
    long int ocv_fix_bias = (ocv_fix + bias_fix);
    float ocv_fl = (double)ocv_fix_bias / cv::pow(2, out_frac);

    std::cout << "hls fix: " << (long int)resultFIX_bias << std::endl;
    std::cout << "ocv fix: " << ocv_fix << std::endl;

    std::cout << "HLS float: " << float_res_fix << std::endl;
    std::cout << "OCV float: " << ocv_fl << std::endl;

    // Error computation
    float diff = abs(abs(float_res_fix) - abs(ocv_fl));
    // std::cout << "HLS fixed point output: " << resultFIX << std::endl;

    std::cout << "Absolute difference: " << diff << std::endl;

    if (diff > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    std::cout << "Test Passed .... !!!" << std::endl;

    return 0;
}
