/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/retinanet.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "usage :" << argv[0] << " <model_name>"
            << " <image_url>" << std::endl;
        abort();
    }

    string kernel = argv[1];
    Mat img = cv::imread(argv[2]);
    if (img.empty()) {
        cerr << "cannot load " << argv[2] << endl;
        abort();
    }

    auto retinanet = vitis::ai::RetinaNet::create(kernel, true);
    if (!retinanet) { // supress coverity complain
        std::cerr <<"create error\n";
        abort();
    }

    /*std::ifstream ifs;
    ifs.open("./data/RetinaNet__input_0_fix.bin");
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    char* buffer = new char[length];
    ifs.read(buffer, length);
    ifs.close();
    Mat bin_mat(800, 800, CV_8SC3, buffer);*/

    int width = retinanet->getInputWidth();
    int height = retinanet->getInputHeight();

    std::cout << "width : " << width
        << " height : " << height
        << std::endl;

    cv::Mat img_resize;

    float w_scale = img.cols * 1.0 / width;
    float h_scale = img.rows * 1.0 / height;
    cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);


    //auto results = retinanet->run(bin_mat);
    auto results = retinanet->run(img_resize);
    for (auto &box : results) {
        rectangle(img_resize, Point(box.x, box.y), Point(box.w, box.h),
            Scalar(rand() % 255, rand() % 255, rand() % 255), 3, 1, 0);
        Point xy;
        xy.x = box.x;
        xy.y = box.y;
        std::ostringstream oss;
        oss << box.score;
        Scalar color = CV_RGB(250,0,0);
        putText(img_resize, oss.str(), xy, FONT_HERSHEY_SIMPLEX, 0.6, color, 1, 8, 0);
    }
    cv::imwrite("out.jpg", img_resize);

    for (auto& box : results) {
      std::cout << "RESULT, "
          << "\t" << (int)(box.x * w_scale)
          << "\t" << (int)(box.y * h_scale)
          << "\t" << (int)(box.w * w_scale)
          << "\t" << (int)(box.h * h_scale)
          << "\t" << box.score
          << "\t" << box.label
          << "\n";
    }
    std::cout << std::endl;

    return 0;
}
