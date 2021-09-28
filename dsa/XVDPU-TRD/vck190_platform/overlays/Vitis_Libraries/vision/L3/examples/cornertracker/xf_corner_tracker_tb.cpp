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
#include "xf_corner_tracker_config.h"

#include "xf_cornertracker_ref.hpp" // OpenCV Reference Code for Corner Tracker Function

#define VIDEO_INPUT 0
#define HLS 0

#if !HLS

#include "xcl2.hpp"
#endif

#define MEMORYALLOC(x) malloc(x)
#define MEMORYFREE(x) free(x)

/* Color Coding */
// kernel returns this type. Packed strcuts on axi ned to be powers-of-2.
typedef struct __rgba {
    unsigned char r, g, b;
    unsigned char a; // can be unused
} rgba_t;
typedef struct __rgb { unsigned char r, g, b; } rgb_t;

typedef cv::Vec<unsigned short, 3> Vec3u;
typedef cv::Vec<unsigned char, 3> Vec3ucpt;

const float powTwo15 = pow(2, 15);
#define THRESHOLD 3.0
#define THRESHOLD_R 3.0
/* color coding */

// custom, hopefully, low cost colorizer.
void getPseudoColorInt(unsigned char pix, float fx, float fy, rgba_t& rgba) {
    // TODO get the normFac from the host as cmdline arg
    const int normFac = 10;

    int y = 127 + (int)(fy * normFac);
    int x = 127 + (int)(fx * normFac);
    if (y > 255) y = 255;
    if (y < 0) y = 0;
    if (x > 255) x = 255;
    if (x < 0) x = 0;

    rgb_t rgb;
    if (x > 127) {
        if (y < 128) {
            // 1 quad
            rgb.r = x - 127 + (127 - y) / 2;
            rgb.g = (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 4 quad
            rgb.r = x - 127;
            rgb.g = 0;
            rgb.b = y - 127;
        }
    } else {
        if (y < 128) {
            // 2 quad
            rgb.r = (127 - y) / 2;
            rgb.g = 127 - x + (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 3 quad
            rgb.r = 0;
            rgb.g = 128 - x;
            rgb.b = y - 127;
        }
    }

    rgba.r = pix * 1 / 2 + rgb.r * 1 / 2;
    rgba.g = pix * 1 / 2 + rgb.g * 1 / 2;
    rgba.b = pix * 1 / 2 + rgb.b * 1 / 2;
    rgba.a = 0;
}

float write_result_to_image_remap_seq(
    cv::Mat imgNext, cv::Mat imgref, cv::Mat glx, cv::Mat gly, int level, char filename[20], char filenamediff[20]) {
    cv::Mat imgRes(imgNext.size(), CV_8U);
    char file1[20], file2[20];
    sprintf(file1, "imgNext_level%d.png", level);
    ;
    sprintf(file2, "result_level%d.png", level);
    ;
    // imwrite(file1, imgNext);
    for (int i = 0; i < imgRes.rows; i++) {
        for (int j = 0; j < imgRes.cols; j++) {
            float indx = (float)j + glx.at<float>(i, j);
            float indy = (float)i + gly.at<float>(i, j);
            // cout << "i = "<<i<<"\t\tj = "<<j<<"\t\tindx = " << indx <<"\t\tindy = "<<indy<<endl;
            // float indx = (float)j;
            // float indy = (float)i;
            int i_indx = (int)indx;
            int i_indy = (int)indy;
            float fx = indx - (float)i_indx;
            float fy = indy - (float)i_indy;
            float iRes;
            if (i_indx > imgNext.cols - 1 || i_indy > imgNext.rows - 1 || i_indx < 0 || i_indy < 0) {
                iRes = 0;
            } else {
                unsigned char i0 = imgNext.at<unsigned char>(i_indy, i_indx);
                unsigned char i1 = imgNext.at<unsigned char>(i_indy, i_indx + 1);
                unsigned char i2 = imgNext.at<unsigned char>(i_indy + 1, i_indx);
                unsigned char i3 = imgNext.at<unsigned char>(i_indy + 1, i_indx + 1);

                iRes = (1 - fx) * (1 - fy) * ((float)i0) + (fx) * (1 - fy) * ((float)i1) +
                       (1 - fx) * (fy) * ((float)i2) + (fx) * (fy) * ((float)i3);
            }
            imgRes.at<unsigned char>(i, j) = (unsigned char)iRes;
        }
    }

    cv::Mat res_absdiff;
    absdiff(imgRes, imgref, res_absdiff);
    float error_percentage = 0;
    for (int t1 = 0; t1 < res_absdiff.rows; t1++) {
        for (int t2 = 0; t2 < res_absdiff.cols; t2++) {
            if (res_absdiff.at<unsigned char>(t1, t2) > 2) error_percentage++;
        }
    }
    error_percentage = error_percentage * 100.0 / ((res_absdiff.rows) * (res_absdiff.cols));
    cv::imwrite(filename, imgRes);
    cv::imwrite(filenamediff, res_absdiff);
    return error_percentage;
} // end write_result_to_image

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "Usage incorrect! Correct usage: ./exe\n<input video or path to input images>\n<no. of "
                "frames>\n<Harris Threshold>\n<No. of frames after which Harris Corners are Reset>\n ");
        return -1;
    }
    char img_name[1000], out_img_name[50], pyr_out_img_name[50], pyr_out_img_name2[50];
    char* path = argv[1];

    std::vector<cv::Point2f> finalPointList, finalPointListCV;
    // TIMER START CODE
    // NOTE: This timer records the time taken to run the entire funciton including the creation of all required
    // variables which increases the reported latency value
    struct timespec begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    finalPointListCV = cornertracker_cv(
        argv[1], atoi(argv[2]),
        atoi(argv[4])); // OpenCV Reference Function, store the final output to compare it to the xf_opencv output

    // TIMER END CODE
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;
    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;
    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;

#if VIDEO_INPUT
    cv::VideoCapture cap;

    std::stringstream imfile;
    imfile.str("");
    imfile << argv[1];
    if (imfile.str() == "") {
        cap.open(0);
        fprintf(stderr, "Invalid input Video\n ");
        if (!cap.isOpened()) {
            fprintf(stderr, "Could not initialize capturing...\n ");
            return 1;
        }
    } else
        cap.open(imfile.str());

    unsigned int imageWidth = (cap.get(CV_CAP_PROP_FRAME_WIDTH));
    unsigned int imageHeight = (cap.get(CV_CAP_PROP_FRAME_HEIGHT));
#else
    cv::Mat frame;
    sprintf(img_name, "%s/im%d.png", path, 0);
    std::cout << "path is" << img_name << std::endl;
    frame = cv::imread(img_name, 1);
    unsigned int imageWidth = frame.cols;
    unsigned int imageHeight = frame.rows;

#endif
    unsigned int harrisThresh = atoi(argv[3]);

    // allocating memory spaces for all the hardware operations
    bool harris_flag = true;
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> inHarris;
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> outHarris;
    unsigned long* listfixed = (unsigned long*)MEMORYALLOC((MAXCORNERS)*8);
    unsigned int* list = (unsigned int*)MEMORYALLOC((MAXCORNERS) * sizeof(unsigned int));
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imagepyr1[NUM_LEVELS];
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imagepyr2[NUM_LEVELS];
    static xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> flow;
    static xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> flow_in;

    unsigned int num_corners = 0;
    unsigned int* params = (unsigned int*)MEMORYALLOC((3) * sizeof(unsigned int));
    // Params &num_corners, harrisThresh, &harris_flag
    params[0] = num_corners; // num_corners
    params[1] = harrisThresh;
    params[2] = harris_flag;

    /*for(int i=0; i<NUM_LEVELS ; i++)
    {
            imagepyr1[i].init(HEIGHT,WIDTH);
            imagepyr2[i].init(HEIGHT,WIDTH);
    }*/
    for (int initlf = 0; initlf < MAXCORNERS; initlf++) {
        list[initlf] = 0;
        listfixed[initlf] = 0;
    }
    flow.init(HEIGHT, WIDTH);
    flow_in.init(HEIGHT, WIDTH);
    inHarris.init(HEIGHT, WIDTH);
    outHarris.init(HEIGHT, WIDTH);
    // initializing flow pointers to 0
    // initializing flow vector with 0s
    cv::Mat init_mat = cv::Mat::zeros(HEIGHT, WIDTH, CV_32SC1);
    flow_in.copyTo((XF_PTSNAME(XF_32UC1, XF_NPPC1)*)init_mat.data);
    flow.copyTo((XF_PTSNAME(XF_32UC1, XF_NPPC1)*)init_mat.data);
    init_mat.release();
    // done
    std::cout << "num of test cases: " << atoi(argv[2]) << "\n";
    cv::Mat im0 = cv::Mat(imageHeight, imageWidth, CV_8UC1, imagepyr1[0].data);
    cv::Mat im1 = cv::Mat(imageHeight, imageWidth, CV_8UC1, imagepyr2[0].data);

#if VIDEO_INPUT
    cv::Mat readVideoImage;
    for (int readn = 0; readn < 1; readn++) {
        cap >> readVideoImage;
        if (readVideoImage.empty()) {
            fprintf(stderr, "im1 is empty\n ");
            break;
        }
    }
    cv::cvtColor(readVideoImage, im1, cv::COLOR_BGR2GRAY);

    cv::VideoWriter video("trackedCorners.avi", CV_FOURCC('M', 'J', 'P', 'G'), 5, cv::Size(imageWidth, imageHeight),
                          true);
#else
    cv::cvtColor(frame, im1, cv::COLOR_BGR2GRAY);

#endif
    ////////////////////	HLS TOP function call	/////////////////

    /////////////////////////////////////// CL ////////////////////////
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_cornertracker");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    char list_name[50], list_fix_name[50];

    for (int i = 0; i < atoi(argv[2]); i++) {
        im1.copyTo(im0);
#if VIDEO_INPUT
        cap >> readVideoImage;
        if (readVideoImage.empty()) {
            fprintf(stderr, "im1 is empty\n ");
            break;
        } else {
            std::cout << "Read frame no. " << i + 1 << std::endl;
        }

        cv::cvtColor(readVideoImage, im1, cv::COLOR_BGR2GRAY);
#else

        sprintf(img_name, "%s/im%d.png", path, i + 1);
        frame = cv::imread(img_name, 1);
        cv::cvtColor(frame, im1, cv::COLOR_BGR2GRAY);

#endif

        std::cout << "***************************************************" << std::endl;
        std::cout << "Test Case no: " << i + 1 << std::endl;

        // Auviz Hardware implementation
        cv::Mat glx(im0.size(), CV_32F, cv::Scalar::all(0)); // flow at each level is updated in this variable
        cv::Mat gly(im0.size(), CV_32F, cv::Scalar::all(0));
        /***********************************************************************************/
        // Setting image sizes for each pyramid level
        int pyr_w[NUM_LEVELS], pyr_h[NUM_LEVELS];
        pyr_h[0] = im0.rows;
        pyr_w[0] = im0.cols;
        for (int lvls = 1; lvls < NUM_LEVELS; lvls++) {
            pyr_h[lvls] = (pyr_h[lvls - 1] + 1) >> 1;
            pyr_w[lvls] = (pyr_w[lvls - 1] + 1) >> 1;
        }

        for (int l = 0; l < NUM_LEVELS; l++) {
            imagepyr1[l].rows = pyr_h[l];
            imagepyr1[l].cols = pyr_w[l];
            imagepyr1[l].size = pyr_h[l] * pyr_w[l];
            imagepyr2[l].rows = pyr_h[l];
            imagepyr2[l].cols = pyr_w[l];
            imagepyr2[l].size = pyr_h[l] * pyr_w[l];
        }
        inHarris.rows = pyr_h[0];
        inHarris.cols = pyr_w[0];
        inHarris.size = pyr_h[0] * pyr_w[0];
        outHarris.rows = pyr_h[0];
        outHarris.cols = pyr_w[0];
        outHarris.size = pyr_h[0] * pyr_w[0];
        inHarris.copyTo(im0.data);
        // CL

        cl::Kernel krnl(program, "cornerTracker");
        cl::Kernel pyr_krnl(program, "pyr_down_accel");

        std::vector<cl::Memory> inBufVec1, outBufVec1, inBufVec2, outBufVec2;
        std::vector<cl::Memory> pyr_inBufVec1, pyr_outBufVec1, pyr_inBufVec2, pyr_outBufVec2;
        std::cout << "\n Harris Execution" << std::endl;

        cl::Buffer imageToDevice1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  (inHarris.rows * inHarris.cols * CH_TYPE), (ap_uint<INPUT_PTR_WIDTH>*)inHarris.data);
        cl::Buffer listfromdevice(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  ((MAXCORNERS) * sizeof(unsigned int)), (unsigned int*)list);
        cl::Buffer params_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, ((3) * sizeof(unsigned int)),
                              (unsigned int*)params);
        std::cout << "\n Harris: Buffers created" << std::endl;
        inBufVec1.push_back(imageToDevice1);
        outBufVec1.push_back(listfromdevice);
        inBufVec2.push_back(params_buf);

        // Copy input vectors to memory
        q.enqueueMigrateMemObjects(inBufVec1, 0);
        q.enqueueMigrateMemObjects(inBufVec2, 0);
        std::cout << "\n Harris: data copied to device" << std::endl;
        // Set the kernel arguments
        krnl.setArg(0, imageToDevice1);
        krnl.setArg(1, listfromdevice);
        krnl.setArg(2, params_buf);
        krnl.setArg(3, inHarris.rows);
        krnl.setArg(4, inHarris.cols);
        std::cout << "\n Harris: args set" << std::endl;

        // Profiling Objects
        cl_ulong start = 0;
        cl_ulong end = 0;
        double diff_prof = 0.0f;
        cl::Event event_sp;

        // Launch the kernel
        std::cout << "\n Harris kernel called" << std::endl;
        q.enqueueTask(krnl, NULL, &event_sp);
        clWaitForEvents(1, (const cl_event*)&event_sp);

        event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        diff_prof = end - start;
        std::cout << (diff_prof / 1000000) << "ms" << std::endl;

        // q.enqueueMigrateMemObjects(outBufVec1,CL_MIGRATE_MEM_OBJECT_HOST);
        q.enqueueMigrateMemObjects(inBufVec2, CL_MIGRATE_MEM_OBJECT_HOST);
        std::cout << "\n Harris Done" << std::endl;
        std::cout << "\n Pyrdown Execution" << std::endl;

        for (int lvl = 0; lvl < NUM_LEVELS - 1; lvl++) {
            cl::Buffer pyr_imageToDevice1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (pyr_h[lvl] * pyr_w[lvl] * CH_TYPE),
                                          (ap_uint<INPUT_PTR_WIDTH>*)imagepyr1[lvl].data);
            cl::Buffer pyr_imageFromDevice1(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                            (pyr_h[lvl + 1] * pyr_w[lvl + 1] * CH_TYPE),
                                            (ap_uint<OUTPUT_PTR_WIDTH>*)imagepyr1[lvl + 1].data);
            cl::Buffer pyr_imageToDevice2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (pyr_h[lvl] * pyr_w[lvl] * CH_TYPE),
                                          (ap_uint<INPUT_PTR_WIDTH>*)imagepyr2[lvl].data);
            cl::Buffer pyr_imageFromDevice2(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                            (pyr_h[lvl + 1] * pyr_w[lvl + 1] * CH_TYPE),
                                            (ap_uint<OUTPUT_PTR_WIDTH>*)imagepyr2[lvl + 1].data);
            std::cout << "\n Pyrdown: Buffers created" << std::endl;
            pyr_inBufVec1.push_back(pyr_imageToDevice1);
            pyr_outBufVec1.push_back(pyr_imageFromDevice1);
            pyr_inBufVec2.push_back(pyr_imageToDevice2);
            pyr_outBufVec2.push_back(pyr_imageFromDevice2);

            // Copy input vectors to memory
            q.enqueueMigrateMemObjects(pyr_inBufVec1, 0);
            q.enqueueMigrateMemObjects(pyr_inBufVec2, 0);
            std::cout << "\n Pyrdown: data copied to device" << std::endl;
            // Set the kernel arguments
            pyr_krnl.setArg(0, pyr_imageToDevice1);
            pyr_krnl.setArg(1, pyr_imageFromDevice1);
            pyr_krnl.setArg(2, pyr_imageToDevice2);
            pyr_krnl.setArg(3, pyr_imageFromDevice2);
            pyr_krnl.setArg(4, pyr_h[lvl]);
            pyr_krnl.setArg(5, pyr_w[lvl]);
            pyr_krnl.setArg(6, pyr_h[lvl + 1]);
            pyr_krnl.setArg(7, pyr_w[lvl + 1]);
            std::cout << "\n Pyrdown Args set" << std::endl;
            std::cout << "\n pyr in ht = " << pyr_h[lvl] << "\n pyr in wd = " << pyr_w[lvl]
                      << "\n pyr out ht = " << pyr_h[lvl + 1] << "\n pyr out wd = " << pyr_w[lvl + 1] << std::endl;

            // Profiling Objects
            cl_ulong pyr_start = 0;
            cl_ulong pyr_end = 0;
            double pyr_diff_prof = 0.0f;
            cl::Event pyr_event_sp;

            // Launch the kernel
            std::cout << "\n Pyrdown kernel called" << std::endl;
            q.enqueueTask(pyr_krnl, NULL, &pyr_event_sp);
            clWaitForEvents(1, (const cl_event*)&pyr_event_sp);

            pyr_event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &pyr_start);
            pyr_event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &pyr_end);
            pyr_diff_prof = pyr_end - pyr_start;
            std::cout << (pyr_diff_prof / 1000000) << "ms" << std::endl;

            q.enqueueMigrateMemObjects(pyr_outBufVec1, CL_MIGRATE_MEM_OBJECT_HOST);
            q.enqueueMigrateMemObjects(pyr_outBufVec2, CL_MIGRATE_MEM_OBJECT_HOST);
            std::cout << "\n Pyrdown data copied to host" << std::endl;
            q.finish();
            std::cout << "\n Pyrdown Execution done" << std::endl;
        }

        // Optical Flow

        std::cout << "\n **********Optical Flow Computation*******************" << std::endl;
        cl::Kernel of_krnl(program, "pyr_dense_optical_flow_accel");
        char name[50], name1[50];
        char in_name[50], in_name1[50];
        cl::Buffer in_img_py1_buf, in_img_py2_buf;
        std::vector<cl::Memory> in_img_py1_Vec, in_img_py2_Vec;

        std::vector<cl::Memory> flow_Vec, flow_in_Vec;
        cl::Buffer flow_buf, flow_in_buf;

        flow_in_buf =
            cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       (pyr_h[NUM_LEVELS - 1] * pyr_w[NUM_LEVELS - 1] * 4), (ap_uint<OUTPUT_PTR_WIDTH>*)flow_in.data);
        flow_in_Vec.push_back(flow_in_buf);
        q.enqueueMigrateMemObjects(flow_in_Vec, 0 /* 0 means from host*/);

        for (int l = NUM_LEVELS - 1; l >= 0; l--) {
            // compute current level height
            int curr_height = pyr_h[l];
            int curr_width = pyr_w[l];

            in_img_py1_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (pyr_h[l] * pyr_w[l] * CH_TYPE), (ap_uint<INPUT_PTR_WIDTH>*)imagepyr1[l].data);
            in_img_py2_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (pyr_h[l] * pyr_w[l] * CH_TYPE), (ap_uint<INPUT_PTR_WIDTH>*)imagepyr2[l].data);
            in_img_py1_Vec.push_back(in_img_py1_buf);
            in_img_py2_Vec.push_back(in_img_py2_buf);

            q.enqueueMigrateMemObjects(in_img_py1_Vec, 0 /* 0 means from host*/);
            q.enqueueMigrateMemObjects(in_img_py2_Vec, 0 /* 0 means from host*/);

            flow_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (pyr_h[l] * pyr_w[l] * 4),
                                  (ap_uint<OUTPUT_PTR_WIDTH>*)flow.data);

            std::cout << "\n Buffers created" << std::endl;
            // compute the flow vectors for the current pyramid level iteratively
            std::cout << "\n  *********OF Computation Level =" << l << "*********" << std::endl;
            flow.init(pyr_h[l], pyr_w[l], 0);
            flow_in.init(pyr_h[l], pyr_w[l], 0);
            for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++) {
                std::cout << "\n  *********OF Computation iteration =" << iterations << "*********" << std::endl;

                bool scale_up_flag = (iterations == 0) && (l != NUM_LEVELS - 1);
                int next_height = (scale_up_flag == 1) ? pyr_h[l + 1] : pyr_h[l];
                int next_width = (scale_up_flag == 1) ? pyr_w[l + 1] : pyr_w[l];
                float scale_in = (next_height - 1) * 1.0 / (curr_height - 1);
                int init_flag = ((iterations == 0) && (l == NUM_LEVELS - 1)) ? 1 : 0;

                if ((iterations == 0) && (l != NUM_LEVELS - 1))
                    flow_in.init(pyr_h[l + 1], pyr_w[l + 1], 0);
                else
                    flow_in.init(pyr_h[l], pyr_w[l], 0);
                std::cout << "\n Data copied from host to device" << std::endl;

                // New way of setting args
                of_krnl.setArg(0, in_img_py1_buf);
                of_krnl.setArg(1, in_img_py2_buf);
                of_krnl.setArg(2, flow_in_buf);
                of_krnl.setArg(3, flow_buf);
                of_krnl.setArg(4, l);
                of_krnl.setArg(5, (int)scale_up_flag);
                of_krnl.setArg(6, scale_in);
                of_krnl.setArg(7, init_flag);
                of_krnl.setArg(8, imagepyr1[l].rows);
                of_krnl.setArg(9, imagepyr1[l].cols);
                of_krnl.setArg(10, imagepyr2[l].rows);
                of_krnl.setArg(11, imagepyr2[l].cols);
                of_krnl.setArg(12, flow_in.rows);
                of_krnl.setArg(13, flow_in.cols);
                of_krnl.setArg(14, flow.rows);
                of_krnl.setArg(15, flow.cols);
                std::cout << "\n kernel args set" << std::endl;
                std::cout << "\n flow_rows =" << flow.rows << "flow_cols=" << flow.cols
                          << "flow_in_rows =" << flow_in.rows << "flow_in_cols =" << flow_in.cols << std::endl;

                cl::Event event_of_sp;

                // Launch the kernel
                q.enqueueTask(of_krnl, NULL, &event_of_sp);
                clWaitForEvents(1, (const cl_event*)&event_of_sp);
                std::cout << "\n level" << l << "calls done" << iterations << std::endl;

                flow_in_buf = flow_buf;

            } // end iterative coptical flow computation

        } // end pyramidal iterative optical flow HLS computation

        q.finish();
        std::cout << "\n OF done" << std::endl;

        std::cout << "\n **********Corner Update Computation*******************" << std::endl;
        cl::Kernel cu_krnl(program, "cornerupdate_accel");

        std::vector<cl::Memory> list_in_Vec, list_fix_Vec;
        cl::Buffer list_in_buf, list_fix_buf;

        list_fix_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, ((MAXCORNERS)*8), listfixed);
        // list_in_Vec.push_back(list_in_buf);
        list_fix_Vec.push_back(list_fix_buf);
        q.enqueueMigrateMemObjects(list_fix_Vec, 0 /* 0 means from host*/);

        cu_krnl.setArg(0, list_fix_buf);
        cu_krnl.setArg(1, listfromdevice);
        cu_krnl.setArg(2, (uint32_t)params[0]);
        cu_krnl.setArg(3, flow_buf);
        cu_krnl.setArg(4, (bool)params[2]);
        cu_krnl.setArg(5, flow.rows);
        cu_krnl.setArg(6, flow.cols);
        std::cout << "\n kernel args set" << std::endl;
        std::cout << "\n flow_rows =" << flow.rows << "flow_cols=" << flow.cols << "num of corners=" << params[0]
                  << "harris_flag=" << params[2] << std::endl;

        cl::Event cu_event_sp;

        // Launch the kernel
        std::cout << "\n Corner Update called" << std::endl;
        q.enqueueTask(cu_krnl, NULL, &cu_event_sp);
        clWaitForEvents(1, (const cl_event*)&cu_event_sp);

        q.enqueueMigrateMemObjects(list_fix_Vec, CL_MIGRATE_MEM_OBJECT_HOST);

        q.enqueueMigrateMemObjects(outBufVec1, CL_MIGRATE_MEM_OBJECT_HOST);

        q.finish();
        std::cout << "\n Corner Update done" << std::endl;

        if (harris_flag == true) harris_flag = false;

        params[2] = harris_flag;

        /////////////////////////////////////// end of CL ////////////////////////

        cv::Mat outputimage; //(im0.rows, im0.cols, CV_8UC3, im1.data);
        cv::cvtColor(im1, outputimage, cv::COLOR_GRAY2BGR);
        std::cout << "\n OF done" << std::endl;
        std::cout << "\n Num of corners = " << params[0] << std::endl;
        for (int li = 0; li < params[0]; li++) {
            unsigned int point = list[li];
            unsigned short row_num = 0;
            unsigned short col_num = 0;

            if (listfixed[li] >> 42 == 0) {
                row_num = (point >> 16) & 0x0000FFFF; //.range(31,16);
                col_num = point & 0x0000FFFF;         //.range(15,0);
                cv::Point2f rmappoint((float)col_num, (float)row_num);
                finalPointList.push_back(rmappoint);
                cv::circle(outputimage, rmappoint, 2, cv::Scalar(0, 0, 255), -1, 8);
            }
        }

#if VIDEO_INPUT
        video.write(outputimage);
#else
        sprintf(out_img_name, "out_img%d.png", i);
        cv::imwrite(out_img_name, outputimage);
#endif
        std::cout << "***************************************************" << std::endl;
        if ((i + 1) % atoi(argv[4]) == 0) {
            params[2] = true;
            params[0] = 0;
            for (int init_list = 0; init_list < MAXCORNERS; init_list++) {
                list[init_list] = 0;
                listfixed[init_list] = 0;
            }
        }
        // releaseing mats and pointers created inside the main for loop
        glx.release();
        gly.release();
        outputimage.release();

        // Print PyrDown outputimage
        for (int kk = 0; kk < NUM_LEVELS; kk++) {
            sprintf(pyr_out_img_name, "pyr1_out_img%d_%d.png", i, kk);
            sprintf(pyr_out_img_name2, "pyr2_out_img%d_%d.png", i, kk);
            xf::cv::imwrite(pyr_out_img_name, imagepyr1[kk]);
            xf::cv::imwrite(pyr_out_img_name2, imagepyr2[kk]);
        }
    }
    im0.data = NULL;
    im1.data = NULL;
#if VIDEO_INPUT
    cap.release();
    video.release();
#endif
    im0.release();
    im1.release();
    MEMORYFREE(list);

    // Error Calculation
    std::vector<double> errVec;
    double ERROR_THRESH = 3;
    uint errCount = 0;
    double errMin = 0, errMax = 0;
    for (int i = 0; i < finalPointList.size(); i++) {
        double minDist = 0;
        for (int j = 0; j < finalPointListCV.size(); j++) {
            double x = finalPointList[i].x - finalPointListCV[j].x;
            double y = finalPointList[i].y - finalPointListCV[j].y;
            double dist = (x * x) + (y * y);
            if (dist < minDist || j == 0) minDist = dist;
        }
        errVec.push_back(minDist);
        if (minDist > ERROR_THRESH) errCount++;
        if (minDist < errMin || i == 0) errMin = minDist;
        if (minDist > errMax || i == 0) errMax = minDist;
    }

    double pixels_above_thresh_per = ((float)errCount * 100.0 / finalPointList.size());
    std::cout << "        Minimum error = " << errMin << std::endl;
    std::cout << "        Maximum error = " << errMax << std::endl;
    std::cout << "        Percentage of points above error threshold = " << pixels_above_thresh_per << std::endl;
    if (pixels_above_thresh_per > 20) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    } else
        std::cout << "Test Passed " << std::endl;

    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;

    return 0;
}
