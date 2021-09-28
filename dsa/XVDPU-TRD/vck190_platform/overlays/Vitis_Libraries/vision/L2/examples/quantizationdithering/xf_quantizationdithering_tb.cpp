/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_quantizationdithering_config.h"
#include <opencv2/core/matx.hpp>

#define MAXVALUE MAXREPRESENTEDVALUE
#if INPUTPIXELDEPTH == 16
#define IN16BIT_EN 1
#else
#define IN16BIT_EN 0
#endif

cv::Mat floyd_steinberg_dithering(cv::Mat image, int scale) {
    cv::Mat new_image, new_image2;
#if GRAY
    new_image.create(cv::Size(image.cols, image.rows), CV_32FC1);
    new_image2.create(cv::Size(image.cols, image.rows), CV_8UC1);
#else
    new_image.create(cv::Size(image.cols, image.rows), CV_32FC3);
    new_image2.create(cv::Size(image.cols, image.rows), CV_8UC3);
#endif
    // new_image = image;
    for (int rowID = 0; rowID < image.rows; rowID++) {
        for (int colID = 0; colID < image.cols; colID++) {
            if (image.channels() == 1) {
#if IN16BIT_EN == 1
                new_image.at<float>(rowID, colID) = (float)image.at<unsigned short>(rowID, colID);
#else
                new_image.at<float>(rowID, colID) = (float)image.at<unsigned char>(rowID, colID);
#endif
            } else if (image.channels() == 3) {
                for (int c = 0; c < 3; c++)
#if IN16BIT_EN == 1
                    new_image.at<cv::Vec3f>(rowID, colID)[c] = (float)image.at<cv::Vec3w>(rowID, colID)[c];
#else
                    new_image.at<cv::Vec3f>(rowID, colID)[c] = (float)image.at<cv::Vec3b>(rowID, colID)[c];
#endif
            }
        }
    }
    cv::imwrite("Sp_input_image.png", new_image);

    int width = image.cols;
    int height = image.rows;
    float old_pix, new_pix;
    float max = MAXVALUE, err_pix;

    for (int rowID = 0; rowID < image.rows; rowID++) {
        for (int colID = 0; colID < image.cols; colID++) {
            if (image.channels() == 1) {
                old_pix = (float)new_image.at<float>(rowID, colID);
                new_pix = round(old_pix * scale / max) * round(max / scale);
                // float orn_pix = (float)image.at<float>(rowID, colID);
                err_pix = old_pix - new_pix;

                if (new_pix >= max) new_pix = (scale - 1) * round(max / scale);
                if (new_pix < 0) new_pix = 0;

                new_image.at<float>(rowID, colID) = round(new_pix / (MAXVALUE / SCALEFACTOR));

                if (colID + 1 < image.cols) {
                    new_image.at<float>(rowID, colID + 1) =
                        ((float)(new_image.at<float>(rowID, colID + 1) + (err_pix * 7 / 16)));
                }
                if ((colID - 1 >= 0) && (rowID + 1 < image.rows)) {
                    new_image.at<float>(rowID + 1, colID - 1) =
                        ((float)(new_image.at<float>(rowID + 1, colID - 1) + (err_pix * 3 / 16)));
                }
                if (rowID + 1 < image.rows) {
                    new_image.at<float>(rowID + 1, colID) =
                        ((float)(new_image.at<float>(rowID + 1, colID) + (err_pix * 5 / 16)));
                }
                if ((colID + 1 < image.cols) && (rowID + 1 < image.rows)) {
                    new_image.at<float>(rowID + 1, colID + 1) =
                        ((float)(new_image.at<float>(rowID + 1, colID + 1) + (err_pix * 1 / 16)));
                }

            } else if (image.channels() == 3) {
                for (int c = 0; c < 3; c++) {
                    old_pix = (float)new_image.at<cv::Vec3f>(rowID, colID)[c];
                    new_pix = round(old_pix * scale / max) * round(max / scale);
                    // float orn_pix = (float)image.at<float>(rowID, colID);
                    err_pix = old_pix - new_pix;

                    if (new_pix >= max) new_pix = (scale - 1) * round(max / scale);
                    if (new_pix < 0) new_pix = 0;

                    new_image.at<cv::Vec3f>(rowID, colID)[c] = round(new_pix / (MAXVALUE / SCALEFACTOR));

                    if (colID + 1 < image.cols) {
                        new_image.at<cv::Vec3f>(rowID, colID + 1)[c] =
                            ((float)(new_image.at<cv::Vec3f>(rowID, colID + 1)[c] + (err_pix * 7 / 16)));
                    }
                    if ((colID - 1 >= 0) && (rowID + 1 < image.rows)) {
                        new_image.at<cv::Vec3f>(rowID + 1, colID - 1)[c] =
                            ((float)(new_image.at<cv::Vec3f>(rowID + 1, colID - 1)[c] + (err_pix * 3 / 16)));
                    }
                    if (rowID + 1 < image.rows) {
                        new_image.at<cv::Vec3f>(rowID + 1, colID)[c] =
                            ((float)(new_image.at<cv::Vec3f>(rowID + 1, colID)[c] + (err_pix * 5 / 16)));
                    }
                    if ((colID + 1 < image.cols) && (rowID + 1 < image.rows)) {
                        new_image.at<cv::Vec3f>(rowID + 1, colID + 1)[c] =
                            ((float)(new_image.at<cv::Vec3f>(rowID + 1, colID + 1)[c] + (err_pix * 1 / 16)));
                    }
                }
            }
        }
    }
    for (int rowID = 0; rowID < image.rows; rowID++) {
        for (int colID = 0; colID < image.cols; colID++) {
            if (image.channels() == 1) {
                new_image2.at<unsigned char>(rowID, colID) = (unsigned char)new_image.at<float>(rowID, colID);
            } else if (image.channels() == 3) {
                for (int c = 0; c < 3; c++)
                    new_image2.at<cv::Vec3b>(rowID, colID)[c] = (unsigned char)new_image.at<cv::Vec3f>(rowID, colID)[c];
            }
        }
    }
    return new_image2;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image> \n");
        return -1;
    }

#if GRAY
    // reading in the color image
    cv::Mat in_img = cv::imread(argv[1], 0);
#else

#if IN16BIT_EN == 1
    cv::Mat in_img = cv::imread(argv[1], -1);
#else
    cv::Mat in_img = cv::imread(argv[1], 1);
#endif

#endif

    if (!in_img.data) {
        fprintf(stderr, "\nImage read failed\n ");
        return -1;
    }
    cv::imwrite("input.png", in_img);

    int in_width, in_height;
    in_width = in_img.cols;
    in_height = in_img.rows;

    cv::Mat out_img, outref_img, error;
#if GRAY
    out_img.create(cv::Size(in_width, in_height), CV_8UC1);
    outref_img.create(cv::Size(in_width, in_height), CV_8UC1);
    error.create(cv::Size(in_width, in_height), CV_32FC1);
    img_fl.create(cv::Size(in_width, in_height), CV_32FC1);
#else
    out_img.create(cv::Size(in_width, in_height), CV_8UC3);
    outref_img.create(cv::Size(in_width, in_height), CV_8UC3);
    error.create(cv::Size(in_width, in_height), CV_32FC3);
#endif

    outref_img = floyd_steinberg_dithering(in_img, SCALEFACTOR);

#if GRAY
#if IN16BIT_EN == 1
    size_t image_in_size_bytes = in_height * in_width * 1 * sizeof(unsigned short);
#else
    size_t image_in_size_bytes = in_height * in_width * 1 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = in_height * in_width * 1 * sizeof(unsigned char);
#else
#if IN16BIT_EN == 1
    size_t image_in_size_bytes = in_height * in_width * 3 * sizeof(unsigned short);
#else
    size_t image_in_size_bytes = in_height * in_width * 3 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = in_height * in_width * 3 * sizeof(unsigned char);
#endif

    ///////////start Opencl //////////////
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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_quantizationdithering");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "quantizationdithering_accel", &err));

    // Allocate the buffers:
    std::vector<cl::Memory> inBufVec, outBufVec;
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevice));
    OCL_CHECK(err, err = krnl.setArg(2, in_height));
    OCL_CHECK(err, err = krnl.setArg(3, in_width));

    /* Copy input vectors to memory */
    OCL_CHECK(err,
              q.enqueueWriteBuffer(imageToDevice,       // buffer on the FPGA
                                   CL_TRUE,             // blocking call
                                   0,                   // buffer offset in bytes
                                   image_in_size_bytes, // Size in bytes
                                   in_img.data));       // Pointer to the data to copy

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Execute the kernel:
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    OCL_CHECK(err, q.enqueueReadBuffer(imageFromDevice, // This buffers data will be read
                                       CL_TRUE,         // blocking call
                                       0,               // offset
                                       image_out_size_bytes,
                                       out_img.data)); // Data will be stored here

    q.finish();
    /////////////////////////////////////// end of CL
    //////////////////////////////////////////

    float err_per;
    cv::absdiff(outref_img, out_img, error);
    xf::cv::analyzeDiff(error, 1, err_per);
    cv::imwrite("hls_output.png", out_img);
    cv::imwrite("output_ref.png", outref_img);

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    }
    std::cout << "Test Passed " << std::endl;

    return 0;
}
