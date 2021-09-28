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

#include "xf_houghlines_config.h"

float sinvalt[360] = {
    0.000000, 0.008727, 0.017452, 0.026177, 0.034899, 0.043619, 0.052336, 0.061049, 0.069756, 0.078459, 0.087156,
    0.095846, 0.104528, 0.113203, 0.121869, 0.130526, 0.139173, 0.147809, 0.156434, 0.165048, 0.173648, 0.182236,
    0.190809, 0.199368, 0.207912, 0.216440, 0.224951, 0.233445, 0.241922, 0.250380, 0.258819, 0.267238, 0.275637,
    0.284015, 0.292372, 0.300706, 0.309017, 0.317305, 0.325568, 0.333807, 0.342020, 0.350207, 0.358368, 0.366501,
    0.374607, 0.382684, 0.390731, 0.398749, 0.406737, 0.414693, 0.422618, 0.430511, 0.438371, 0.446198, 0.453991,
    0.461749, 0.469472, 0.477159, 0.484810, 0.492424, 0.499980, 0.507539, 0.515038, 0.522499, 0.529920, 0.537300,
    0.544639, 0.551937, 0.559193, 0.566407, 0.573577, 0.580703, 0.587786, 0.594823, 0.601815, 0.608762, 0.615662,
    0.622515, 0.629321, 0.636079, 0.642788, 0.649448, 0.656059, 0.662620, 0.669131, 0.675591, 0.681999, 0.688355,
    0.694659, 0.700910, 0.707107, 0.713251, 0.719340, 0.725375, 0.731354, 0.737278, 0.743145, 0.748956, 0.754710,
    0.760406, 0.766045, 0.771625, 0.777146, 0.782609, 0.788011, 0.793354, 0.798636, 0.803857, 0.809017, 0.814116,
    0.819152, 0.824127, 0.829038, 0.833886, 0.838671, 0.843392, 0.848048, 0.852641, 0.857168, 0.861629, 0.866026,
    0.870356, 0.874620, 0.878817, 0.882948, 0.887011, 0.891007, 0.894934, 0.898794, 0.902585, 0.906308, 0.909961,
    0.913545, 0.917060, 0.920505, 0.923879, 0.927184, 0.930417, 0.933580, 0.936672, 0.939692, 0.942641, 0.945518,
    0.948323, 0.951056, 0.953717, 0.956305, 0.958820, 0.961261, 0.963630, 0.965926, 0.968147, 0.970295, 0.972370,
    0.974370, 0.976296, 0.978147, 0.979924, 0.981627, 0.983255, 0.984807, 0.986285, 0.987688, 0.989016, 0.990268,
    0.991445, 0.992546, 0.993572, 0.994522, 0.995396, 0.996195, 0.996917, 0.997564, 0.998135, 0.998629, 0.999048,
    0.999391, 0.999657, 0.999848, 0.999962, 0.999980, 0.999962, 0.999848, 0.999657, 0.999391, 0.999048, 0.998629,
    0.998135, 0.997564, 0.996917, 0.996195, 0.995396, 0.994522, 0.993572, 0.992546, 0.991445, 0.990268, 0.989016,
    0.987688, 0.986285, 0.984807, 0.983255, 0.981627, 0.979924, 0.978147, 0.976296, 0.97437,  0.97237,  0.970295,
    0.968147, 0.965926, 0.96363,  0.961261, 0.95882,  0.956305, 0.953717, 0.951056, 0.948323, 0.945518, 0.942641,
    0.939692, 0.936672, 0.93358,  0.930417, 0.927184, 0.923879, 0.920505, 0.91706,  0.913545, 0.909961, 0.906308,
    0.902585, 0.898794, 0.894934, 0.891007, 0.887011, 0.882948, 0.878817, 0.87462,  0.870356, 0.866026, 0.861629,
    0.857168, 0.852641, 0.848048, 0.843392, 0.838671, 0.833886, 0.829038, 0.824127, 0.819152, 0.814116, 0.809017,
    0.803857, 0.798636, 0.793354, 0.788011, 0.782609, 0.777146, 0.771625, 0.766045, 0.760406, 0.75471,  0.748956,
    0.743145, 0.737278, 0.731354, 0.725375, 0.71934,  0.713251, 0.707107, 0.70091,  0.694659, 0.688355, 0.681999,
    0.675591, 0.669131, 0.66262,  0.656059, 0.649448, 0.642788, 0.636079, 0.629321, 0.622515, 0.615662, 0.608762,
    0.601815, 0.594823, 0.587786, 0.580703, 0.573577, 0.566407, 0.559193, 0.551937, 0.544639, 0.5373,   0.52992,
    0.522499, 0.515038, 0.507539, 0.5,      0.492424, 0.48481,  0.477159, 0.469472, 0.461749, 0.453991, 0.446198,
    0.438371, 0.430511, 0.422618, 0.414693, 0.406737, 0.398749, 0.390731, 0.382684, 0.374607, 0.366501, 0.358368,
    0.350207, 0.34202,  0.333807, 0.325568, 0.317305, 0.309017, 0.300706, 0.292372, 0.284015, 0.275637, 0.267238,
    0.258819, 0.25038,  0.241922, 0.233445, 0.224951, 0.21644,  0.207912, 0.199368, 0.190809, 0.182236, 0.173648,
    0.165048, 0.156434, 0.147809, 0.139173, 0.130526, 0.121869, 0.113203, 0.104528, 0.095846, 0.087156, 0.078459,
    0.069756, 0.061049, 0.052336, 0.043619, 0.034899, 0.026177, 0.017452, 0.008727};
float cosvalt[360] = {
    0.999980,  0.999962,  0.999848,  0.999657,  0.999391,  0.999048,  0.998629,  0.998135,  0.997564,  0.996917,
    0.996195,  0.995396,  0.994522,  0.993572,  0.992546,  0.991445,  0.990268,  0.989016,  0.987688,  0.986285,
    0.984807,  0.983255,  0.981627,  0.979924,  0.978147,  0.976296,  0.97437,   0.97237,   0.970295,  0.968147,
    0.965926,  0.96363,   0.961261,  0.95882,   0.956305,  0.953717,  0.951056,  0.948323,  0.945518,  0.942641,
    0.939692,  0.936672,  0.93358,   0.930417,  0.927184,  0.923879,  0.920505,  0.91706,   0.913545,  0.909961,
    0.906308,  0.902585,  0.898794,  0.894934,  0.891007,  0.887011,  0.882948,  0.878817,  0.87462,   0.870356,
    0.866026,  0.861629,  0.857168,  0.852641,  0.848048,  0.843392,  0.838671,  0.833886,  0.829038,  0.824127,
    0.819152,  0.814116,  0.809017,  0.803857,  0.798636,  0.793354,  0.788011,  0.782609,  0.777146,  0.771625,
    0.766045,  0.760406,  0.75471,   0.748956,  0.743145,  0.737278,  0.731354,  0.725375,  0.71934,   0.713251,
    0.707107,  0.70091,   0.694659,  0.688355,  0.681999,  0.675591,  0.669131,  0.66262,   0.656059,  0.649448,
    0.642788,  0.636079,  0.629321,  0.622515,  0.615662,  0.608762,  0.601815,  0.594823,  0.587786,  0.580703,
    0.573577,  0.566407,  0.559193,  0.551937,  0.544639,  0.5373,    0.52992,   0.522499,  0.515038,  0.507539,
    0.5,       0.492424,  0.48481,   0.477159,  0.469472,  0.461749,  0.453991,  0.446198,  0.438371,  0.430511,
    0.422618,  0.414693,  0.406737,  0.398749,  0.390731,  0.382684,  0.374607,  0.366501,  0.358368,  0.350207,
    0.34202,   0.333807,  0.325568,  0.317305,  0.309017,  0.300706,  0.292372,  0.284015,  0.275637,  0.267238,
    0.258819,  0.25038,   0.241922,  0.233445,  0.224951,  0.21644,   0.207912,  0.199368,  0.190809,  0.182236,
    0.173648,  0.165048,  0.156434,  0.147809,  0.139173,  0.130526,  0.121869,  0.113203,  0.104528,  0.095846,
    0.087156,  0.078459,  0.069756,  0.061049,  0.052336,  0.043619,  0.034899,  0.026177,  0.017452,  0.008727,
    0.000000,  -0.008727, -0.017452, -0.026177, -0.034899, -0.043619, -0.052336, -0.061049, -0.069756, -0.078459,
    -0.087156, -0.095846, -0.104528, -0.113203, -0.121869, -0.130526, -0.139173, -0.147809, -0.156434, -0.165048,
    -0.173648, -0.182236, -0.190809, -0.199368, -0.207912, -0.216440, -0.224951, -0.233445, -0.241922, -0.250380,
    -0.258819, -0.267238, -0.275637, -0.284015, -0.292372, -0.300706, -0.309017, -0.317305, -0.325568, -0.333807,
    -0.342020, -0.350207, -0.358368, -0.366501, -0.374607, -0.382684, -0.390731, -0.398749, -0.406737, -0.414693,
    -0.422618, -0.430511, -0.438371, -0.446198, -0.453991, -0.461749, -0.469472, -0.477159, -0.484810, -0.492424,
    -0.499980, -0.507539, -0.515038, -0.522499, -0.529920, -0.537300, -0.544639, -0.551937, -0.559193, -0.566407,
    -0.573577, -0.580703, -0.587786, -0.594823, -0.601815, -0.608762, -0.615662, -0.622515, -0.629321, -0.636079,
    -0.642788, -0.649448, -0.656059, -0.662620, -0.669131, -0.675591, -0.681999, -0.688355, -0.694659, -0.700910,
    -0.707107, -0.713251, -0.719340, -0.725375, -0.731354, -0.737278, -0.743145, -0.748956, -0.754710, -0.760406,
    -0.766045, -0.771625, -0.777146, -0.782609, -0.788011, -0.793354, -0.798636, -0.803857, -0.809017, -0.814116,
    -0.819152, -0.824127, -0.829038, -0.833886, -0.838671, -0.843392, -0.848048, -0.852641, -0.857168, -0.861629,
    -0.866026, -0.870356, -0.874620, -0.878817, -0.882948, -0.887011, -0.891007, -0.894934, -0.898794, -0.902585,
    -0.906308, -0.909961, -0.913545, -0.917060, -0.920505, -0.923879, -0.927184, -0.930417, -0.933580, -0.936672,
    -0.939692, -0.942641, -0.945518, -0.948323, -0.951056, -0.953717, -0.956305, -0.958820, -0.961261, -0.963630,
    -0.965926, -0.968147, -0.970295, -0.972370, -0.974370, -0.976296, -0.978147, -0.979924, -0.981627, -0.983255,
    -0.984807, -0.986285, -0.987688, -0.989016, -0.990268, -0.991445, -0.992546, -0.993572, -0.994522, -0.995396,
    -0.996195, -0.996917, -0.997564, -0.998135, -0.998629, -0.999048, -0.999391, -0.999657, -0.999848, -0.999962};

struct LinePolar {
    float rho;
    float angle;
};

struct hough_cmp_gt {
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const { return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2); }
    const int* aux;
};

void xiHoughLinesstandard(cv::Mat& img,
                          std::vector<cv::Vec2f>& lines,
                          float rho,
                          float theta,
                          int threshold,
                          int linesMax,
                          int maxtheta,
                          int mintheta) {
    int i, j;
    float irho = 1 / rho;

    CV_Assert(img.type() == CV_8UC1);

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;
    double max_theta;
    double min_theta;

    if (maxtheta > 0) max_theta = (CV_PI * maxtheta) / 180;
    if (mintheta > 0) min_theta = (CV_PI * mintheta) / 180;

    if (max_theta < min_theta) {
        CV_Error(cv::Error::StsBadArg, "max_theta must be greater than min_theta");
    }
    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound((sqrt(width * width + height * height)) / rho);

    cv::AutoBuffer<int> _accum((numangle + 2) * (numrho + 2));
    std::vector<int> _sort_buf;
    cv::AutoBuffer<float> _tabSin(numangle);
    cv::AutoBuffer<float> _tabCos(numangle);
    int* accum = _accum;
    float *tabSin = _tabSin, *tabCos = _tabCos;

    memset(accum, 0, sizeof(accum[0]) * (numangle + 2) * (numrho + 2));

    float thetaind = (theta * 180) / CV_PI;

    int ang = 2 * mintheta;

    for (int n = 0; n < numangle; ang += (2 * thetaind), n++) {
        tabSin[n] = (float)sinvalt[ang] * irho;
        tabCos[n] = (float)cosvalt[ang] * irho;
    }

    float temp[360], tempsinval[360], tempcosval[360];
    for (int i = 0; i < 360; i++) {
        temp[i] = 0.0;
        tempsinval[i] = 0.0;
        tempcosval[i] = 0.0;
    }
    // stage 1. fill accumulator
    float r1;

    int hei = (height / 2);
    int wdt = (width / 2);

    for (int ki = 0; ki < numangle; ki++) {
        tempsinval[ki] = (-hei) * tabSin[ki];
        tempcosval[ki] = (-wdt) * tabCos[ki];
    }

    for (i = 0; i < height; i++) // i -->row
    {
        for (int ki = 0; ki < numangle; ki++) {
            if (i > 0) tempsinval[ki] = tempsinval[ki] + tabSin[ki];
        }

        for (j = 0; j < width; j++) // j-->col
        {
            for (int n = 0; n < numangle; n++) {
                int rho1;

                if (j == 0) {
                    r1 = tempcosval[n] + tempsinval[n];
                } else {
                    r1 = temp[n] + tabCos[n];
                }

                temp[n] = r1;

                rho1 = cvRound(r1) + ((numrho) / 2);

                if (image[(i) * (step) + (j)] != 0) {
                    accum[(n + 1) * (numrho + 2) + rho1 + 1]++;
                }
            }
        }
    }

    // stage 2. find local maximums
    for (int r = 0; r < numrho; r++)
        for (int n = 0; n < numangle; n++) {
            int base = (n + 1) * (numrho + 2) + r + 1;
            if (accum[base] > threshold && accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                _sort_buf.push_back(base);
        }

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1. / (numrho + 2);

    for (i = 0; i < linesMax; i++) {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx * scale) - 1;
        int r = idx - (n + 1) * (numrho + 2) - 1;
        line.rho = (r - (numrho)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(cv::Vec2f(line.rho, line.angle));
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img, ocv_ref;
    cv::Mat in_gray, in_gray1, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for output images:
    ocv_ref.create(in_gray.rows, in_gray.cols, in_gray.depth());
    out_img.create(in_gray.rows, in_gray.cols, in_gray.depth());

    cv::Mat dst, cdst, crefdst, crefxi, crefcv;
    cv::Canny(in_gray, dst, 50, 200, 3);

    cvtColor(dst, cdst, cv::COLOR_GRAY2BGR);
    cvtColor(dst, crefdst, cv::COLOR_GRAY2BGR);
    cvtColor(dst, crefxi, cv::COLOR_GRAY2BGR);
    cvtColor(dst, crefcv, cv::COLOR_GRAY2BGR);

    // std::vector<float> outputrho(LINESMAX);
    // std::vector<float> outputtheta(LINESMAX);

    float* outputrho = (float*)malloc(LINESMAX * sizeof(float));
    float* outputtheta = (float*)malloc(LINESMAX * sizeof(float));

    short threshold = 20;
    short maxlines = LINESMAX;

    int height = in_gray.rows;
    int width = in_gray.cols;

    houghlines_accel((ap_uint<PTR_WIDTH>*)dst.data, threshold, LINESMAX, outputrho, outputtheta, height, width);

    // Running the reference function:
    std::vector<cv::Vec2f> linesxi;

    float thetaval = (THETASTEP / 2.0);
    float angleref = (CV_PI * thetaval) / 180;

    xiHoughLinesstandard(dst, linesxi, RHOSTEP, angleref, threshold, maxlines, MAXTHETA,
                         MINTHETA); // FLOATING POINT Reference code

    FILE* fpre1 = fopen("HLSOUT.txt", "w");
    FILE* fpre2 = fopen("FLOATREFOUT.txt", "w");

    for (size_t i = 0; i < linesxi.size(); i++) {
        fprintf(fpre1, "%f %f\n", outputrho[i], outputtheta[i]);
        fprintf(fpre2, "%f %f\n", linesxi[i][0], linesxi[i][1]);
    }

    fclose(fpre1);
    fclose(fpre2);

    // Results verification:
    int count = 0;
    float successper = 0.0;

    for (size_t i = 0; i < linesxi.size(); i++) {
        for (size_t j = 0; j < linesxi.size(); j++) {
            float difftheta, diffrho;

            diffrho = outputrho[j] - linesxi[i][0];
            difftheta = fabs(outputtheta[j] - linesxi[i][1]);

            if (diffrho <= 1.0 && difftheta <= 0.01f) {
                count++;
                break;
            }
        }
    }

    successper = ((float)count / linesxi.size()) * 100;

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tSuccess = " << successper << "%" << std::endl;
    std::cout << "\tNumber of matched lines = " << count << std::endl;

    if (successper < 90) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

#if 0
	int heiby2 = (height/2);
	int wdtby2 = (width/2);

	for( size_t i = 0; i < linesxi.size(); i++ )
	{
		float rho = linesxi[i][0], theta = linesxi[i][1];
		cv::Point pt1xi, pt2xi;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1xi.x = cvRound(x0 + 750*(-b) + (wdtby2));
		pt1xi.y = cvRound(y0 + 750*(a) + (heiby2));
		pt2xi.x = cvRound(x0 - 750*(-b) + (wdtby2));
		pt2xi.y = cvRound(y0 - 750*(a) + (heiby2));
		cv::line( crefxi, pt1xi, pt2xi, cv::Scalar(0,0,255), 1, CV_AA);
	}

	for( size_t i = 0; i < linesxi.size(); i++ )
	{
		float rho = outputrho[i], theta = outputtheta[i];
		cv::Point pt1xi, pt2xi;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1xi.x = cvRound(x0 + 1500*(-b) + (wdtby2)) ;
		pt1xi.y = cvRound(y0 + 1500*(a) + (heiby2));
		pt2xi.x = cvRound(x0 - 1500*(-b) + (wdtby2));
		pt2xi.y = cvRound(y0 - 1500*(a) + (heiby2));
		cv::line( cdst, pt1xi, pt2xi, cv::Scalar(0,0,255), 1, CV_AA);
	}

	// Write down the output images:
	cv::imwrite("outhls.png",cdst);
	cv::imwrite("out_floatref.png",crefxi);

#endif

    return 0;
}
