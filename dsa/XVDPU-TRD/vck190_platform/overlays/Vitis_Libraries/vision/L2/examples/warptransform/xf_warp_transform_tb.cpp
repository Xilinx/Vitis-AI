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
#include "xf_warp_transform_config.h"

// Changing transformation matrix dimensions with transform Affine 2x3,
// Perspecitve 3x3
#if TRANSFORM_TYPE == 1
#define TRMAT_DIM2 3
#define TRMAT_DIM1 3
#else
#define TRMAT_DIM2 3
#define TRMAT_DIM1 2
#endif

// Random Number generator limits
#define M_NUMI1 1
#define M_NUMI2 20

// Image operations and transformation matrix input format
typedef float image_oper;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    cv::RNG rng;
    std::vector<float> R(9);
    cv::Mat _transformation_matrix(TRMAT_DIM1, TRMAT_DIM2, CV_32FC1);
    cv::Mat _transformation_matrix_2(TRMAT_DIM1, TRMAT_DIM2, CV_32FC1);

#if TRANSFORM_TYPE == 1
    cv::Point2f src_p[4];
    cv::Point2f dst_p[4];
    src_p[0] = cv::Point2f(0.0f, 0.0f);
    src_p[1] = cv::Point2f(WIDTH - 1, 0.0f);
    src_p[2] = cv::Point2f(WIDTH - 1, HEIGHT - 1);
    src_p[3] = cv::Point2f(0.0f, HEIGHT - 1);
    //	  to points
    dst_p[0] = cv::Point2f(rng.uniform(int(M_NUMI1), int(M_NUMI2)), rng.uniform(int(M_NUMI1), int(M_NUMI2)));
    dst_p[1] = cv::Point2f(WIDTH - rng.uniform(int(M_NUMI1), int(M_NUMI2)), rng.uniform(int(M_NUMI1), int(M_NUMI2)));
    dst_p[2] =
        cv::Point2f(WIDTH - rng.uniform(int(M_NUMI1), int(M_NUMI2)), HEIGHT - rng.uniform(int(M_NUMI1), int(M_NUMI2)));
    dst_p[3] = cv::Point2f(rng.uniform(int(M_NUMI1), int(M_NUMI2)), HEIGHT - rng.uniform(int(M_NUMI1), int(M_NUMI2)));

    _transformation_matrix = cv::getPerspectiveTransform(dst_p, src_p);
    cv::Mat transform_mat = _transformation_matrix;
#else
    cv::Point2f src_p[3];
    cv::Point2f dst_p[3];
    src_p[0] = cv::Point2f(0.0f, 0.0f);
    src_p[1] = cv::Point2f(WIDTH - 1, 0.0f);
    src_p[2] = cv::Point2f(0.0f, HEIGHT - 1);
    //	  to points
    dst_p[0] = cv::Point2f(rng.uniform(int(M_NUMI1), int(M_NUMI2)), rng.uniform(int(M_NUMI1), int(M_NUMI2)));
    dst_p[1] = cv::Point2f(WIDTH - rng.uniform(int(M_NUMI1), int(M_NUMI2)), rng.uniform(int(M_NUMI1), int(M_NUMI2)));
    dst_p[2] = cv::Point2f(rng.uniform(int(M_NUMI1), int(M_NUMI2)), HEIGHT - rng.uniform(int(M_NUMI1), int(M_NUMI2)));

    _transformation_matrix = cv::getAffineTransform(dst_p, src_p);
    cv::Mat transform_mat = _transformation_matrix;
#endif

    int i = 0, j = 0;

    std::cout << "INFO: Transformation Matrix is:";
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
#if TRANSFORM_TYPE == 1
            R[i * 3 + j] = image_oper(transform_mat.at<double>(i, j));
            _transformation_matrix_2.at<image_oper>(i, j) = image_oper(transform_mat.at<double>(i, j));
#else
            if (i == 2) {
                R[i * 3 + j] = 0;
            } else {
                R[i * 3 + j] = image_oper(transform_mat.at<double>(i, j));
                _transformation_matrix_2.at<image_oper>(i, j) = image_oper(transform_mat.at<double>(i, j));
            }
#endif
            std::cout << R[i * 3 + j] << " ";
        }
        std::cout << "\n";
    }

    cv::Mat image_input, image_output, diff_img;

// Reading in the image:
#if GRAY
    image_input = cv::imread(argv[1], 0);
#else
    image_input = cv::imread(argv[1], 1);
#endif
    if (!image_input.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Allocate memory for the output images:
    image_output.create(image_input.rows, image_input.cols, image_input.type());
    diff_img.create(image_input.rows, image_input.cols, image_input.type());

// OpenCL section:
#if GRAY
    size_t image_in_size_bytes = image_input.rows * image_input.cols * sizeof(unsigned char);

#else
    size_t image_in_size_bytes = image_input.rows * image_input.cols * 3 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = image_in_size_bytes;
    size_t vec_in_size_bytes = R.size() * sizeof(float);

    std::cout << "INFO: In size =" << image_in_size_bytes << "out size = " << image_out_size_bytes << std::endl;
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
    ;
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_warptransform");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    std::cout << "INFO: Program Created" << std::endl;
    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "warptransform_accel", &err));
    std::cout << "INFO: Kernel Created" << std::endl;
    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inVec(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    std::cout << "INFO: Buffers Created" << std::endl;
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inVec));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, image_input.rows));
    OCL_CHECK(err, err = kernel.setArg(4, image_input.cols));

    std::cout << "INFO: Args Set" << std::endl;

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            image_input.data,    // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inVec,      // buffer on the FPGA
                                       CL_TRUE,           // blocking call
                                       0,                 // buffer offset in bytes
                                       vec_in_size_bytes, // Size in bytes
                                       R.data(),          // Pointer to the data to copy
                                       nullptr, &event));

    std::cout << "INFO: Data transferred from host to device" << std::endl;

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));
    std::cout << "INFO: Kernel Called" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            image_output.data, // Data will be stored here
                            nullptr, &event);

    std::cout << "INFO: Data copied from device to host" << std::endl;
    // Clean up:
    queue.finish();
    cv::imwrite("output.png", image_output);

    // OpenCV reference:
    cv::Mat opencv_image;
#if GRAY
    opencv_image.create(image_input.rows, image_input.cols, CV_8UC1);
#else
    opencv_image.create(image_input.rows, image_input.cols, CV_8UC3);
#endif

    for (int I1 = 0; I1 < opencv_image.rows; I1++) {
        for (int J1 = 0; J1 < opencv_image.cols; J1++) {
#if GRAY
            opencv_image.at<ap_uint8_t>(I1, J1) = 0;
#else
            opencv_image.at<cv::Vec3b>(I1, J1) = 0;
#endif
        }
    }

#if TRANSFORM_TYPE == 1
#if INTERPOLATION == 1
    cv::warpPerspective(image_input, opencv_image, _transformation_matrix_2,
                        cv::Size(image_input.cols, image_input.rows), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                        cv::BORDER_TRANSPARENT, 80);
#else
    cv::warpPerspective(image_input, opencv_image, _transformation_matrix_2,
                        cv::Size(image_input.cols, image_input.rows), cv::INTER_NEAREST + cv::WARP_INVERSE_MAP,
                        cv::BORDER_TRANSPARENT, 80);
#endif
#else
#if INTERPOLATION == 1
    cv::warpAffine(image_input, opencv_image, _transformation_matrix_2, cv::Size(image_input.cols, image_input.rows),
                   cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT, 80);
#else
    cv::warpAffine(image_input, opencv_image, _transformation_matrix_2, cv::Size(image_input.cols, image_input.rows),
                   cv::INTER_NEAREST + cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT, 80);
#endif
#endif

    cv::imwrite("opencv_output.png", opencv_image);

    float err_per;

    cv::absdiff(image_output, opencv_image, diff_img);

    xf::cv::analyzeDiff(diff_img, 0, err_per);
}
