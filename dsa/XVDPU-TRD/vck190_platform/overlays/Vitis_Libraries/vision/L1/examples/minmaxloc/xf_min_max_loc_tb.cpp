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
#include "xf_min_max_loc_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray, in_conv;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

/*  convert to gray  */
// cvtColor(in_img,in_gray,CV_BGR2GRAY);

/*  convert to 16S type  */
#if T_8U
    in_img.convertTo(in_conv, CV_8UC1);
#elif T_16U
    in_img.convertTo(in_conv, CV_16UC1);
#elif T_16S
    in_img.convertTo(in_conv, CV_16SC1);
#elif T_32S
    in_img.convertTo(in_conv, CV_32SC1);
#endif

    double cv_minval = 0, cv_maxval = 0;
    cv::Point cv_minloc, cv_maxloc;
    int height = in_img.rows;
    int width = in_img.cols;

    /////////  OpenCV reference  ///////
    cv::minMaxLoc(in_conv, &cv_minval, &cv_maxval, &cv_minloc, &cv_maxloc, cv::noArray());

    int32_t min_value, max_value;
    uint16_t _min_locx, _min_locy, _max_locx, _max_locy;

    ////////  Call the Top function ///
    min_max_loc_accel((ap_uint<PTR_WIDTH>*)in_conv.data, min_value, max_value, _min_locx, _min_locy, _max_locx,
                      _max_locy, height, width);

    /////// OpenCV output ////////
    std::cout << "OCV-Minvalue = " << cv_minval << std::endl;
    std::cout << "OCV-Maxvalue = " << cv_maxval << std::endl;
    std::cout << "OCV-Min Location.x = " << cv_minloc.x << "  OCV-Min Location.y = " << cv_minloc.y << std::endl;
    std::cout << "OCV-Max Location.x = " << cv_maxloc.x << "  OCV-Max Location.y = " << cv_maxloc.y << std::endl;

    /////// Kernel output ////////
    std::cout << "HLS-Minvalue = " << min_value << std::endl;
    std::cout << "HLS-Maxvalue = " << max_value << std::endl;
    std::cout << "HLS-Min Location.x = " << _min_locx << "  HLS-Min Location.y = " << _min_locy << std::endl;
    std::cout << "HLS-Max Location.x = " << _max_locx << "  HLS-Max Location.y = " << _max_locy << std::endl;

    /////// printing the difference in min and max, values and locations of both OpenCV and Kernel function /////
    printf("Difference in Minimum value: %d \n", (cv_minval - min_value));
    printf("Difference in Maximum value: %d \n", (cv_maxval - max_value));
    printf("Difference in Minimum value location: (%d,%d) \n", (cv_minloc.y - _min_locy), (cv_minloc.x - _min_locx));
    printf("Difference in Maximum value location: (%d,%d) \n", (cv_maxloc.y - _max_locy), (cv_maxloc.x - _max_locx));

    if (((cv_minloc.y - _min_locy) > 1) | ((cv_minloc.x - _min_locx) > 1)) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    }

    return 0;
}
