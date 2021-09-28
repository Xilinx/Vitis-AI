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
#include "opencv2/features2d.hpp"

#include "xf_bfmatcher_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <image1 path> <image2 path> \n");
        return -1;
    }

    cv::Mat in_img_1, in_img_2, out_img, ocv_ref;
    cv::Mat diff;

    // reading in the image
    in_img_1 = cv::imread(argv[1], 0);
    in_img_2 = cv::imread(argv[2], 0);

    if (in_img_1.data == NULL) {
        fprintf(stderr, "Cannot open image 1 at %s\n", argv[1]);
        return 0;
    }

    if (in_img_2.data == NULL) {
        fprintf(stderr, "Cannot open image 2 at %s\n", argv[2]);
        return 0;
    }

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<std::array<uint8_t, 32> > descriptors1, descriptors2;
    std::vector<cv::DMatch> good_matches;

    // reading desc and keypoints //
    int nkp1;
    int nkp2;
    const float ratio_thresh = 0.7f;

    cv::Mat descriptors1_cv, descriptors2_cv;
    // Initiate ORB detector
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    // find the keypoints and descriptors with ORB
    detector->detect(in_img_1, keypoints1);
    detector->detect(in_img_2, keypoints2);

    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    extractor->compute(in_img_1, keypoints1, descriptors1_cv);
    extractor->compute(in_img_2, keypoints2, descriptors2_cv);

    nkp1 = descriptors1_cv.rows;
    nkp2 = descriptors2_cv.rows;

    descriptors1.resize(nkp1);
    for (int i = 0; i < nkp1; i++) {
        for (int j = 0; j < 32; j++) {
            descriptors1[i][j] = descriptors1_cv.at<unsigned char>(i, j);
        }
    }

    descriptors2.resize(nkp2);
    for (int i = 0; i < nkp2; i++) {
        for (int j = 0; j < 32; j++) {
            descriptors2[i][j] = descriptors2_cv.at<unsigned char>(i, j);
        }
    }
    // end of read //

    // Opencv reference
    cv::Ptr<cv::DescriptorMatcher> matcher_cv;
    std::vector<std::vector<cv::DMatch> > knn_matches_cv;

    matcher_cv = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    matcher_cv->knnMatch(descriptors1_cv, descriptors2_cv, knn_matches_cv, 2);

    // adding lowe's ratio test for comparison
    std::vector<cv::DMatch> good_matches_cv;
    for (size_t i = 0; i < knn_matches_cv.size(); i++) {
        if (knn_matches_cv[i][0].distance < ratio_thresh * knn_matches_cv[i][1].distance) {
            good_matches_cv.push_back(knn_matches_cv[i][0]);
        }
    }
    //	End of OpenCV reference

    ///////////////////   HLS module  /////////////////

    int16_t tmp_good_matches[nkp1];

    bfmatcher_accel((ap_uint<INPUT_PTR_WIDTH>*)descriptors1.data(), (ap_uint<INPUT_PTR_WIDTH>*)descriptors2.data(),
                    (ap_int<OUTPUT_PTR_WIDTH>*)tmp_good_matches, nkp1, nkp2, ratio_thresh);

    int rev_matches = 0;
    for (int i = 0; i < nkp1; i++) {
        if ((int)tmp_good_matches[i] >= 0) rev_matches++;
    }

    good_matches.resize(rev_matches);
    rev_matches = 0;
    for (int i = 0; i < nkp1; i++) {
        if ((int)tmp_good_matches[i] >= 0) {
            good_matches[rev_matches].queryIdx = i;
            good_matches[rev_matches].trainIdx = (int)tmp_good_matches[i];
            rev_matches++;
        }
    }

    //-- Draw matches
    cv::Mat img_matches, img_matches_cv;
    cv::drawMatches(in_img_1, keypoints1, in_img_2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(in_img_1, keypoints1, in_img_2, keypoints2, good_matches_cv, img_matches_cv, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    cv::imwrite("Good_Matches.png", img_matches);
    cv::imwrite("Good_Matches_cv.png", img_matches_cv);

    // validate functionality
    int wrong_matches = 0;
    for (int i = 0; i < rev_matches; i++) {
        if ((good_matches[i].queryIdx != good_matches_cv[i].queryIdx) ||
            (good_matches[i].trainIdx != good_matches_cv[i].trainIdx)) {
            printf("wrong matches found in the validation!");
            return -1;
        }
    }

    std::cout << "Test passed successfully!" << std::endl;

    return 0;
}
