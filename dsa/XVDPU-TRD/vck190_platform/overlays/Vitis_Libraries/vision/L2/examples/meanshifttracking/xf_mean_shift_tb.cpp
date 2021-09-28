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
#include "xf_mean_shift_config.h"

#include "xcl2.hpp"

#define _PROFILE_ 0

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr,
                "Missed input arguments. Usage: <executable> <path to input video file or image path> <Number of "
                "objects to be tracked> \n");
        return -1;
    }

    char* path = argv[1];

#if VIDEO_INPUT
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) // check if we succeeded
    {
        fprintf(stderr, "ERROR: Cannot open the video file\n ");
        return -1;
    }
#endif
    uint8_t no_objects = atoi(argv[2]);

    uint16_t* c_x = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* c_y = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* h_x = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* h_y = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* tlx = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* tly = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* brx = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* bry = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* track = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* obj_width = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* obj_height = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* dx = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));
    uint16_t* dy = (uint16_t*)malloc(XF_MAX_OBJECTS * sizeof(uint16_t));

    for (int i = 0; i < no_objects; i++) {
        dx[i] = 0;
        dy[i] = 0;
    }

    // object loop, for reading input to the object
    for (uint16_t i = 0; i < no_objects; i++) {
        h_x[i] = WIDTH_MST[i] / 2;
        h_y[i] = HEIGHT_MST[i] / 2;
        c_x[i] = X1[i] + h_x[i];
        c_y[i] = Y1[i] + h_y[i];

        obj_width[i] = WIDTH_MST[i];
        obj_height[i] = HEIGHT_MST[i];

        tlx[i] = X1[i];
        tly[i] = Y1[i];
        brx[i] = c_x[i] + h_x[i];
        bry[i] = c_y[i] + h_y[i];
        track[i] = 1;
    }

    cv::Mat frame, image;
    int no_of_frames = TOTAL_FRAMES;
    char nm[1000];

#if VIDEO_INPUT
    cap >> frame;
#else
    sprintf(nm, "%s/img%d.png", path, 1);
    frame = cv::imread(nm, 1);
#endif

    int height = frame.rows;
    int width = frame.cols;

    cl_int err;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_mean_shift");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "mean_shift_accel", &err));

    for (int f_no = 1; f_no <= no_of_frames; f_no++) {
        if (f_no > 1) {
#if VIDEO_INPUT
            cap >> frame;
#else
            sprintf(nm, "%s/img%d.png", path, f_no);
            frame = cv::imread(nm, 1);
#endif
        }

        if (frame.empty()) {
            fprintf(stderr, "no image!\n");
            break;
        }
        frame.copyTo(image);

        // convert to four channels with a dummy alpha channel for 32-bit data transfer
        cvtColor(image, image, cv::COLOR_BGR2RGBA);

        // set the status of the frame, set as '0' for the first frame
        uint8_t frame_status = 1;
        if (f_no - 1 == 0) frame_status = 0;

        uint8_t no_of_iterations = 4;

        /////////////////////////////////////// CL ////////////////////////
        OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * 4), NULL, &err));
        OCL_CHECK(err,
                  cl::Buffer tlxToDevice(context, CL_MEM_READ_ONLY, no_objects * sizeof(unsigned short), NULL, &err));
        OCL_CHECK(err,
                  cl::Buffer tlyToDevice(context, CL_MEM_READ_ONLY, no_objects * sizeof(unsigned short), NULL, &err));
        OCL_CHECK(err, cl::Buffer objHeightToDevice(context, CL_MEM_READ_ONLY, no_objects * sizeof(unsigned short),
                                                    NULL, &err));
        OCL_CHECK(err, cl::Buffer objWidthToDevice(context, CL_MEM_READ_ONLY, no_objects * sizeof(unsigned short), NULL,
                                                   &err));
        OCL_CHECK(err,
                  cl::Buffer dxFromDevice(context, CL_MEM_WRITE_ONLY, no_objects * sizeof(unsigned short), NULL, &err));
        OCL_CHECK(err,
                  cl::Buffer dyFromDevice(context, CL_MEM_WRITE_ONLY, no_objects * sizeof(unsigned short), NULL, &err));
        OCL_CHECK(err, cl::Buffer trackFromDevice(context, CL_MEM_READ_WRITE, no_objects * sizeof(unsigned short), NULL,
                                                  &err));

        // Set the kernel arguments
        OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
        OCL_CHECK(err, err = krnl.setArg(1, tlxToDevice));
        OCL_CHECK(err, err = krnl.setArg(2, tlyToDevice));
        OCL_CHECK(err, err = krnl.setArg(3, objHeightToDevice));
        OCL_CHECK(err, err = krnl.setArg(4, objWidthToDevice));
        OCL_CHECK(err, err = krnl.setArg(5, dxFromDevice));
        OCL_CHECK(err, err = krnl.setArg(6, dyFromDevice));
        OCL_CHECK(err, err = krnl.setArg(7, trackFromDevice));
        OCL_CHECK(err, err = krnl.setArg(8, frame_status));
        OCL_CHECK(err, err = krnl.setArg(9, no_objects));
        OCL_CHECK(err, err = krnl.setArg(10, no_of_iterations));
        OCL_CHECK(err, err = krnl.setArg(11, height));
        OCL_CHECK(err, err = krnl.setArg(12, width));

        // initiate write to clBuffer
        OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * width * 4), image.data));
        OCL_CHECK(err, q.enqueueWriteBuffer(tlxToDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), tlx));
        OCL_CHECK(err, q.enqueueWriteBuffer(tlyToDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), tly));
        OCL_CHECK(err,
                  q.enqueueWriteBuffer(objHeightToDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), obj_height));
        OCL_CHECK(err,
                  q.enqueueWriteBuffer(objWidthToDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), obj_width));
        OCL_CHECK(err, q.enqueueWriteBuffer(trackFromDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), track));

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

        OCL_CHECK(err, q.enqueueReadBuffer(dxFromDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), dx));
        OCL_CHECK(err, q.enqueueReadBuffer(dyFromDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), dy));
        OCL_CHECK(err, q.enqueueReadBuffer(trackFromDevice, CL_TRUE, 0, no_objects * sizeof(unsigned short), track));
        q.finish();
        /////////////////////////////////////// end of CL ////////////////////////

        std::cout << "frame " << f_no << ":" << std::endl;
        for (int k = 0; k < no_objects; k++) {
            c_x[k] = dx[k];
            c_y[k] = dy[k];
            tlx[k] = c_x[k] - h_x[k];
            tly[k] = c_y[k] - h_y[k];
            brx[k] = c_x[k] + h_x[k];
            bry[k] = c_y[k] + h_y[k];

            std::cout << " " << c_x[k] << " " << c_y[k] << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
