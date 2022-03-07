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
#include "xf_pyr_dense_optical_flow_config.h"

#include "opencv2/video.hpp"
#include "xf_lkdensepyrof_ref.hpp"
#include <time.h>

#include "xcl2.hpp"
/* Color Coding */
// kernel returns this type. Packed strcuts on axi ned to be powers-of-2.
typedef struct __rgba {
    IN_TYPE r, g, b;
    IN_TYPE a; // can be unused
} rgba_t;
typedef struct __rgb { IN_TYPE r, g, b; } rgb_t;

typedef cv::Vec<unsigned short, 3> Vec3u;
typedef cv::Vec<IN_TYPE, 3> Vec3ucpt;

const float powTwo15 = pow(2, 15);
#define THRESHOLD 3.0
#define THRESHOLD_R 3.0
/* color coding */
#define NORM_FAC 10
// custom, hopefully, low cost colorizer.
void getPseudoColorInt(IN_TYPE pix, float fx, float fy, rgba_t& rgba) {
    // TODO get the normFac from the host as cmdline arg
    // const int normFac = 10;

    int y = 127 + (int)(fy * NORM_FAC);
    int x = 127 + (int)(fx * NORM_FAC);
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

void pyrof_hw(cv::Mat im0,
              cv::Mat im1,
              cv::Mat flowUmat,
              cv::Mat flowVmat,
              xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>& flow,
              xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>& flow_in,
              xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> mat_imagepyr1[NUM_LEVELS],
              xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> mat_imagepyr2[NUM_LEVELS],
              int pyr_h[NUM_LEVELS],
              int pyr_w[NUM_LEVELS]) {
    for (int i = 0; i < pyr_h[0]; i++) {
        for (int j = 0; j < pyr_w[0]; j++) {
            mat_imagepyr1[0].write(i * pyr_w[0] + j, im0.data[i * pyr_w[0] + j]);
            mat_imagepyr2[0].write(i * pyr_w[0] + j, im1.data[i * pyr_w[0] + j]);
        }
    }
    // creating image pyramid

    /////////////////////////////////////// CL ////////////////////////
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    // Queue Creation
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_pyr_dense_optical_flow");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    cl::Kernel krnl(program, "pyr_down_accel");
    fprintf(stdout, "\n *********Pyr Down Execution*********\n");

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < NUM_LEVELS - 1; i++) {
            std::vector<cl::Memory> inBufVec, outBufVec;
            cl::Buffer imageToDevice, imageFromDevice;

            if (j == 0)
                imageToDevice = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (pyr_h[i] * pyr_w[i] * CH_TYPE), mat_imagepyr1[i].data);
            else
                imageToDevice = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (pyr_h[i] * pyr_w[i] * CH_TYPE), mat_imagepyr2[i].data);

            if (j == 0)
                imageFromDevice = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                             (pyr_h[i + 1] * pyr_w[i + 1] * CH_TYPE), mat_imagepyr1[i + 1].data);
            else
                imageFromDevice = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                             (pyr_h[i + 1] * pyr_w[i + 1] * CH_TYPE), mat_imagepyr2[i + 1].data);
            fprintf(stdout, "\n CL buffer created\n");

            inBufVec.push_back(imageToDevice);
            outBufVec.push_back(imageFromDevice);

            /* Copy input vectors to memory */
            q.enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/);

            fprintf(stdout, "\n data copied to host\n");
            // Set the kernel arguments
            krnl.setArg(0, imageToDevice);
            krnl.setArg(1, imageFromDevice);
            krnl.setArg(2, pyr_h[i]);
            krnl.setArg(3, pyr_w[i]);
            krnl.setArg(4, pyr_h[i + 1]);
            krnl.setArg(5, pyr_w[i + 1]);
            fprintf(stdout, "\n Kernel args set\n");
            // Profiling Objects
            cl_ulong start = 0;
            cl_ulong end = 0;
            double diff_prof = 0.0f;
            cl::Event event_sp;

            // Launch the kernel
            q.enqueueTask(krnl, NULL, &event_sp);
            clWaitForEvents(1, (const cl_event*)&event_sp);

            event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            diff_prof = end - start;
            std::cout << (diff_prof / 1000000) << "ms" << std::endl;

            q.enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
            q.finish();

            fprintf(stdout, "\n%d image  %d level pyrdown done\n", j, i);
            char out_name[20];
            sprintf(out_name, "xf_image_%d%d.png", j, i);
            if (j == 0)
                xf::cv::imwrite(out_name, mat_imagepyr1[i + 1]);
            else
                xf::cv::imwrite(out_name, mat_imagepyr2[i + 1]);
        }
        fprintf(stdout, "\n One image done\n");
    }

    fprintf(stdout, "\n *********Pyr Down Done*********\n");

    bool flag_flowin = 1;
    flow.init(pyr_h[NUM_LEVELS - 1], pyr_w[NUM_LEVELS - 1], 0);

    flow_in.init(pyr_h[NUM_LEVELS - 1], pyr_w[NUM_LEVELS - 1], 0);

    cl::Kernel krnl2(program, "pyr_dense_optical_flow_accel");
    fprintf(stdout, "\n *********Starting OF Computation*********\n");
    char name[50], name1[50];
    char in_name[50], in_name1[50];
    cl::Buffer in_img_py1_buf, in_img_py2_buf;
    std::vector<cl::Memory> in_img_py1_Vec, in_img_py2_Vec;

    std::vector<cl::Memory> flow_Vec, flow_in_Vec;
    cl::Buffer flow_buf, flow_in_buf;

    flow_in_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (pyr_h[NUM_LEVELS - 1] * pyr_w[NUM_LEVELS - 1] * 4), flow_in.data);
    flow_in_Vec.push_back(flow_in_buf);
    q.enqueueMigrateMemObjects(flow_in_Vec, 0 /* 0 means from host*/);

    for (int l = NUM_LEVELS - 1; l >= 0; l--) {
        // compute current level height
        int curr_height = pyr_h[l];
        int curr_width = pyr_w[l];

        in_img_py1_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (pyr_h[l] * pyr_w[l] * CH_TYPE),
                                    mat_imagepyr1[l].data);
        in_img_py2_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (pyr_h[l] * pyr_w[l] * CH_TYPE),
                                    mat_imagepyr2[l].data);
        in_img_py1_Vec.push_back(in_img_py1_buf);
        in_img_py2_Vec.push_back(in_img_py2_buf);

        q.enqueueMigrateMemObjects(in_img_py1_Vec, 0 /* 0 means from host*/);
        q.enqueueMigrateMemObjects(in_img_py2_Vec, 0 /* 0 means from host*/);

        flow_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (pyr_h[l] * pyr_w[l] * 4), flow.data);

        fprintf(stdout, "\nBuffers created\n");

        // compute the flow vectors for the current pyramid level iteratively
        fprintf(stdout, "\n *********OF Computation Level = %d*********\n", l);
        flow.init(pyr_h[l], pyr_w[l], 0);
        flow_in.init(pyr_h[l], pyr_w[l], 0);
        for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++) {
            fprintf(stdout, "\n *********OF Computation iteration = %d*********\n", iterations);

            bool scale_up_flag = (iterations == 0) && (l != NUM_LEVELS - 1);
            int next_height = (scale_up_flag == 1) ? pyr_h[l + 1] : pyr_h[l];
            int next_width = (scale_up_flag == 1) ? pyr_w[l + 1] : pyr_w[l];
            float scale_in = (next_height - 1) * 1.0 / (curr_height - 1);
            int init_flag = ((iterations == 0) && (l == NUM_LEVELS - 1)) ? 1 : 0;

            if ((iterations == 0) && (l != NUM_LEVELS - 1))
                flow_in.init(pyr_h[l + 1], pyr_w[l + 1], 0);
            else
                flow_in.init(pyr_h[l], pyr_w[l], 0);

            fprintf(stdout, "\nData copied from host to device\n");

            // New way of setting args
            krnl2.setArg(0, in_img_py1_buf);
            krnl2.setArg(1, in_img_py2_buf);
            krnl2.setArg(2, flow_in_buf);
            krnl2.setArg(3, flow_buf);
            krnl2.setArg(4, l);
            krnl2.setArg(5, (int)scale_up_flag);
            krnl2.setArg(6, scale_in);
            krnl2.setArg(7, init_flag);
            krnl2.setArg(8, mat_imagepyr1[l].rows);
            krnl2.setArg(9, mat_imagepyr1[l].cols);
            krnl2.setArg(10, mat_imagepyr2[l].rows);
            krnl2.setArg(11, mat_imagepyr2[l].cols);
            krnl2.setArg(12, flow_in.rows);
            krnl2.setArg(13, flow_in.cols);
            krnl2.setArg(14, flow.rows);
            krnl2.setArg(15, flow.cols);
            fprintf(stdout, "\nkernel args set\n");

            cl::Event event_of_sp;

            // Launch the kernel
            q.enqueueTask(krnl2, NULL, &event_of_sp);
            clWaitForEvents(1, (const cl_event*)&event_of_sp);
            fprintf(stdout, "\n%d level %d calls done\n", l, iterations);

            flow_in_buf = flow_buf;

        } // end iterative coptical flow computation

    } // end pyramidal iterative optical flow HLS computation
    flow_Vec.push_back(flow_buf);
    q.enqueueMigrateMemObjects(flow_Vec, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    // FILE* fp_flow = fopen("flow_op.txt", "w");
    // write output flow vectors to Mat after splitting the bits.
    for (int i = 0; i < pyr_h[0]; i++) {
        for (int j = 0; j < pyr_w[0]; j++) {
            unsigned int tempcopy = 0;
            {
                // tempcopy = *(flow.data + i*pyr_w[0] + j);
                tempcopy = flow.read(i * pyr_w[0] + j);
            }
            // fprintf(fp_flow, "%u\n", tempcopy);
            short splittemp1 = (tempcopy >> 16);
            short splittemp2 = (0x0000FFFF & tempcopy);

            TYPE_FLOW_TYPE* uflow = (TYPE_FLOW_TYPE*)&splittemp1;
            TYPE_FLOW_TYPE* vflow = (TYPE_FLOW_TYPE*)&splittemp2;

            flowUmat.at<float>(i, j) = (float)*uflow;
            flowVmat.at<float>(i, j) = (float)*vflow;
        }
    }
    // fclose(fp_flow);
    return;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage incorrect! Correct usage: ./exe <current image> <next image>\n");
        return -1;
    }
    // allocating memory spaces for all the hardware operations
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imagepyr1[NUM_LEVELS];
    static xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imagepyr2[NUM_LEVELS];
    static xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> flow;
    static xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> flow_in;

    // initializing flow pointers to 0
    // initializing flow vector with 0s
    cv::Mat init_mat = cv::Mat::zeros(HEIGHT, WIDTH, CV_32SC1);
    flow_in.copyTo((XF_PTSNAME(XF_32UC1, XF_NPPC1)*)init_mat.data);
    flow.copyTo((XF_PTSNAME(XF_32UC1, XF_NPPC1)*)init_mat.data);
    init_mat.release();

    cv::Mat im0, im1;

    // Read the file
    im0 = cv::imread(argv[1], 0);
    im1 = cv::imread(argv[2], 0);
    if (im0.empty()) {
        fprintf(stderr, "Loading image 1 failed, exiting!!\n");
        return -1;
    } else if (im1.empty()) {
        fprintf(stderr, "Loading image 2 failed, exiting!!\n");
        return -1;
    }

    // OpenCV Implementation
    cv::Mat glx_cv = cv::Mat::zeros(im0.size(), CV_32F);
    cv::Mat gly_cv = cv::Mat::zeros(im0.size(), CV_32F);
#if __XF_BENCHMARK
    int rows = im0.rows, cols = im0.cols;

    std::vector<cv::Point2f> pt0, pt1;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) pt0.push_back(cv::Point2f(j, i));
    const cv::Size winSize(WINSIZE_OFLOW, WINSIZE_OFLOW);
    std::vector<uchar> status;
    std::vector<float> err;

    // TIMER START CODE
    struct timespec begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    cv::calcOpticalFlowPyrLK(im0, im1, pt0, pt1, status, err, winSize, NUM_LEVELS);

    // TIMER END CODE
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;
    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    for (int i = 0; i < pt0.size(); i++) {
        if (status[i]) {
            glx_cv.at<float>((int)pt0[i].y, (int)pt0[i].x) = pt1[i].x - pt0[i].x;
            gly_cv.at<float>((int)pt0[i].y, (int)pt0[i].x) = pt1[i].y - pt0[i].y;
        }
    }
#else
    refOpticalFlow(im0, im1, glx_cv, gly_cv);
#endif

    // Auviz Hardware implementation
    cv::Mat glx(im0.size(), CV_32F, cv::Scalar::all(0)); // flow at each level is updated in this variable
    cv::Mat gly(im0.size(), CV_32F, cv::Scalar::all(0));
    /***********************************************************************************/
    // Setting image sizes for each pyramid level
    int pyr_w[NUM_LEVELS], pyr_h[NUM_LEVELS];
    pyr_h[0] = im0.rows;
    pyr_w[0] = im0.cols;
    for (int lvls = 1; lvls < NUM_LEVELS; lvls++) {
        pyr_w[lvls] = (pyr_w[lvls - 1] + 1) >> 1;
        pyr_h[lvls] = (pyr_h[lvls - 1] + 1) >> 1;
    }

    for (int i = 0; i < NUM_LEVELS; i++) {
        imagepyr1[i].init(pyr_h[i], pyr_w[i]);
        imagepyr2[i].init(pyr_h[i], pyr_w[i]);
    }
    flow.init(HEIGHT, WIDTH);
    flow_in.init(HEIGHT, WIDTH);

    // call the hls optical flow implementation
    pyrof_hw(im0, im1, glx, gly, flow, flow_in, imagepyr1, imagepyr2, pyr_h, pyr_w);

    // output file names for the current case
    char colorout_filename[20] = "flow_image.png";
    char colorout_filename_cv[20] = "flow_image_cv.png";

    // Color code the flow vectors on original image
    Vec3ucpt color_px;

    cv::Mat color_code_img;
    color_code_img.create(im0.size(), CV_8UC3);
    for (int rc = 0; rc < im0.rows; rc++) {
        for (int cc = 0; cc < im0.cols; cc++) {
            rgba_t colorcodedpx;
            getPseudoColorInt(im0.at<unsigned char>(rc, cc), glx.at<float>(rc, cc), gly.at<float>(rc, cc),
                              colorcodedpx);
            color_px = Vec3ucpt(colorcodedpx.b, colorcodedpx.g, colorcodedpx.r);
            color_code_img.at<Vec3ucpt>(rc, cc) = color_px;
        }
    }
    cv::imwrite(colorout_filename, color_code_img);
    color_code_img.release();

    cv::Mat color_code_img_cv;
    color_code_img_cv.create(im0.size(), CV_8UC3);
    for (int rc = 0; rc < im0.rows; rc++) {
        for (int cc = 0; cc < im0.cols; cc++) {
            rgba_t colorcodedpx;
            getPseudoColorInt(im0.at<unsigned char>(rc, cc), glx_cv.at<float>(rc, cc), gly_cv.at<float>(rc, cc),
                              colorcodedpx);
            color_px = Vec3ucpt(colorcodedpx.b, colorcodedpx.g, colorcodedpx.r);
            color_code_img_cv.at<Vec3ucpt>(rc, cc) = color_px;
        }
    }
    cv::imwrite(colorout_filename_cv, color_code_img_cv);
    color_code_img_cv.release();
// end color coding

#if __XF_BENCHMARK == 0
    const int ERROR_THRESH = 5;
    cv::Mat glx_diff, gly_diff, glx_diff_thresh, gly_diff_thresh;
    cv::absdiff(glx, glx_cv, glx_diff);
    cv::absdiff(gly, gly_cv, gly_diff);

    double u_min, u_max, v_min, v_max, min, max;
    cv::minMaxLoc(glx_diff, &u_min, &u_max);
    cv::minMaxLoc(gly_diff, &v_min, &v_max);
    min = (u_min < v_min) ? u_min : v_min;
    max = (u_max > v_max) ? u_max : v_max;

    glx_diff = glx_diff * NORM_FAC;
    gly_diff = gly_diff * NORM_FAC;
    threshold(glx_diff, glx_diff_thresh, ERROR_THRESH, 1, cv::THRESH_BINARY_INV);
    threshold(gly_diff, gly_diff_thresh, ERROR_THRESH, 1, cv::THRESH_BINARY_INV);
    cv::Mat valid_out = glx_diff_thresh.mul(gly_diff_thresh);

    int pixels_under_thresh = cv::countNonZero(valid_out);
    int total_pixels = im0.rows * im0.cols;
    int pixels_above_thresh = total_pixels - pixels_under_thresh;
    float pixels_above_thresh_per = ((float)pixels_above_thresh) * 100.0 / total_pixels;

    std::cout << "        Minimum error in intensity = " << min << std::endl;
    std::cout << "        Maximum error in intensity = " << max << std::endl;
    std::cout << "        Percentage of pixels above error threshold = " << pixels_above_thresh_per << std::endl;
    if (pixels_above_thresh_per > 20) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    } else
        std::cout << "Test Passed " << std::endl;
#else
    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;
#endif

    // releaseing mats and pointers created inside the main for loop
    glx.release();
    gly.release();
    glx_cv.release();
    gly_cv.release();
    im0.release();
    im1.release();
    return 0;
}
