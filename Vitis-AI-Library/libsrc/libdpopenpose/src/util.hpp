/*
 * Copyright 2019 Xilinx Inc.
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
#ifndef OPENPOSE_UTIL_HPP
#define OPENPOSE_UTIL_HPP

#include <xilinx/openpose/openpose.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace cv;

namespace xilinx { namespace openpose {
using Peak = std::tuple<int, float, cv::Point2f>;
using Peaks = vector<Peak>;
using AllPeaks = vector<Peaks>;
using Candidate = tuple<int, int, float, float>;
using Connection = tuple<int, int, float, int, int>;
using AllConnection = vector<Connection>;

static vector<vector<int>> limbSeq = {{0,1}, {1,2}, {2,3}, {3,4}, {1,5}, {5,6}, {6,7}, {1,8}, \
			       {8,9}, {9,10}, {1,11}, {11,12}, {12,13}};
static vector<vector<int>> mapIdx = {{15,16}, {17,18}, {19,20}, {21,22}, {23,24}, {25,26}, {27,28}, {29,30}, \
		               {31,32}, {33,34}, {35,36}, {37,38}, {39,40}};
bool isThreeInConnection(const vector<Connection> &connections, int index){
    for(size_t i = 0; i < connections.size(); ++i){
	if(index == get<3>(connections[i])) return true;
    }
    return false;
}

bool isFourInConnection(const vector<Connection> &connections, int index){
    for(size_t i = 0; i < connections.size(); ++i){
	if(index == get<4>(connections[i])) return true;
    }
    return false;
}

void find_peak_neon(Mat ori_img, Peaks &peaks, int &idx) {
    float32x4_t comp_f32 = vdupq_n_f32(0.1);
    Mat gas_img;
    vector<uint32_t> r{0,0,0,0};
    GaussianBlur(ori_img, gas_img, Size(3,3), 3);
    float* gas_data = (float*)gas_img.data;
    for(int y = 1; y < gas_img.rows - 3; y++) {
        for(int x = 1; x < gas_img.cols - 3; x = x + 4){
            float32x4_t src_f32_self = vld1q_f32(gas_data + y*gas_img.cols + x);
            uint32x4_t condition_result = vcgtq_f32(src_f32_self, comp_f32);
            vst1q_u32(r.data(), condition_result);
            for(size_t i = 0; i < r.size(); i++) {
                if(r[i] == 0) {
                    continue;
                }
                auto value = gas_data[y*gas_img.cols + x + i];
                if(value >= gas_data[y*gas_img.cols + x + i - 1]
                && value >= gas_data[(y - 1)*gas_img.cols + x + i]
                && value >= gas_data[y*gas_img.cols + x + i + 1]
                && value >= gas_data[(y + 1) *gas_img.cols + x + i]
                    ){
                        peaks.emplace_back(++idx, ori_img.at<float>(y, x + i), Point(x + i, y));
                }
            }
        }
    }
}

void findLines(Mat resimg, const vector<Mat>& pafs, const AllPeaks& all_peaks, \
	       vector<AllConnection>& connection_all, vector<int>& special_k){
    vector<Connection> connection;
    int mid_num = 10;
    for(size_t k = 0; k < mapIdx.size(); ++k){
	Mat score_midx = pafs[mapIdx[k][0] -15];
	Mat score_midy = pafs[mapIdx[k][1] -15];
	Peaks candA = all_peaks[limbSeq[k][0] ];
	Peaks candB = all_peaks[limbSeq[k][1] ];
	size_t nA = candA.size();
	size_t nB = candB.size();
	vector<float> vec;
	vec.reserve(2);
	if(!candA.empty() && !candB.empty()){
	    vector<Candidate> connection_candidate;
	    for(size_t i = 0; i < candA.size(); ++i){
	        for(size_t j = 0; j < candB.size(); ++j){
		    vec[0] = get<2>(candA[i]).x - get<2>(candB[j]).x;
		    vec[1] = get<2>(candA[i]).y - get<2>(candB[j]).y;
		    float norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
		    vector<Point> points;
		    for(int a = 0; a < mid_num; ++a){
			points.emplace_back(Point(int(round(get<2>(candA[i]).x -  a * vec[0] / (mid_num-1))), \
				          int(round(get<2>(candA[i]).y -  a * vec[1] / (mid_num-1)))) );
		    }
		    vec[0] = vec[0] / norm;
		    vec[1] = vec[1] / norm;
      		    vector<float> vec_x;
		    vector<float> vec_y;
		    vector<float> score_midpts;
		    float sum = 0;
		    int lencir = 0;
		    for(size_t b = 0; b < points.size(); ++b){
			 vec_x.emplace_back(score_midx.at<float>(points[b].y, points[b].x));
			 vec_y.emplace_back(score_midy.at<float>(points[b].y, points[b].x));
			 score_midpts.emplace_back(abs(vec_x[b] * vec[0] + vec_y[b] * vec[1]));
			 sum += score_midpts[b];
			 if(score_midpts[b] > 0.05) lencir++;
		    }
		    float score_with_dist_prior = sum / score_midpts.size()  \
						  + min(0.5 * resimg.cols / norm - 1, 0.0);
		    bool cirterion1 = lencir > 0.8 * score_midpts.size();
		    bool cirterion2 = score_with_dist_prior > 0;
		    if(cirterion1 && cirterion2){
  		  	 connection_candidate.emplace_back(i, j, score_with_dist_prior, \
		 	 score_with_dist_prior + get<1>(candA[i]) + get<1>(candB[j]) );
		    }
	         }
	    }
	    std::sort(connection_candidate.begin(), connection_candidate.end(),
	    [](const tuple<int, int, float,float >& lhs,const tuple<int, int, float,float >& rhs) {
	    return get<2>(lhs) > get<2>(rhs);  });
	    connection.clear();
	    for(size_t c = 0; c < connection_candidate.size(); ++c){
	        int i = get<0>(connection_candidate[c]);
	        int j = get<1>(connection_candidate[c]);
	        float s = get<2>(connection_candidate[c]);
	        if( !isThreeInConnection(connection, i) && !isFourInConnection(connection, j)){
		   connection.emplace_back(get<0>(candA[i]), get<0>(candB[j]), s, i, j);
		   if(connection.size() >= min(nA, nB) ) break;
		}
	    }
  	    connection_all.emplace_back(connection);
 	}
	else {
	    special_k.emplace_back(k);
	    connection.clear();
	    connection_all.emplace_back(connection);
	}
    }
}

std::vector<std::vector<OpenPoseResult::PosePoint>> getPoses(const AllPeaks& all_peaks, vector<AllConnection>& connection_all, vector<int>& special_k){
    vector<vector<int>> subset(0, vector<int>(16, -1));
    Peaks candidate;
    for(auto peaks : all_peaks){
        for(auto peak : peaks){
            candidate.emplace_back(peak);
        }
    }
    for(size_t k = 0; k < mapIdx.size(); ++k){
        if(find(special_k.begin(), special_k.end(), k) == special_k.end()){
            int indexA = limbSeq[k][0];
            int indexB = limbSeq[k][1];
            for(size_t i = 0; i < connection_all[k].size(); ++i){
                int found = 0;
                int partA = get<0>(connection_all[k][i]);
                int partB = get<1>(connection_all[k][i]);
                vector<int> subset_idx(2,-1);
                for(size_t j = 0; j < subset.size(); ++j){
                    if(subset[j][indexA] == partA ||
                       subset[j][indexB] == partB ){
                        subset_idx[found] = j;
                        found += 1;
                    }
                }
                if(found == 1){
                    int j = subset_idx[0];
                    if(subset[j][indexB] != partB){
                        subset[j][indexB] = partB;
                        subset[j][15] += 1;
                        subset[j][14] += get<0>(candidate[partA]) + get<2>(connection_all[k][i]);
                    }
                }
                else if(found == 2){
                    int j1 = subset_idx[0];
                    int j2 = subset_idx[1];
                    vector<int> membership(14, 0);
                    for(size_t a = 0; a < membership.size(); ++a){
                        int x = subset[j1][a] >= 0 ? 1 : 0;
                        int y = subset[j2][a] >= 0 ? 1 : 0;
                        membership[a] = x + y;
                    }
                    if(find(membership.begin(), membership.end(), 2) == membership.end()){
                        for(size_t a = 0; a < subset.size() - 2; ++a){
                            subset[j1][a] += (subset[j2][a] + 1);
                        }
                        for(size_t a = subset.size() - 2; a < subset.size(); ++a){
                            subset[j1][a] += subset[j2][a];
                        }
                        subset[j1][13] += get<2>(connection_all[k][i]);
                    }
                    else {
                        subset[j1][indexB] = partA;
                        subset[j1][15] += 1;
                        subset[j1][14] += get<0>(candidate[partB]) + get<2>(connection_all[k][i]);
                    }
                }
                else if ( found == 0 && k < 14){
                    vector<int> row(16, -1);
                    row[indexA] = partA;
                    row[indexB] = partB;
                    row[15] = 2;
                    row[14] = get<0>(candidate[partA]) + get<0>(candidate[partB]) + get<2>(connection_all[k][i]);
                    subset.emplace_back(row);
                }
            }
        }
    }
    for(size_t i = 0; i < subset.size(); ++i){
        for( size_t j = 0; j < subset[i].size(); ++j){
        }
        if(subset[i][15] < 4 || subset[i][14]/subset[i][15] < 0.4){
            subset.erase(subset.begin() + i);
            --i;
        }
    }
    OpenPoseResult::PosePoint posePoint;
    std::vector<std::vector<OpenPoseResult::PosePoint>> poses(subset.size()+1, vector<OpenPoseResult::PosePoint>(14,posePoint));
    for(size_t i = 0; i < subset.size(); ++i){
        for(int j = 0; j < 14; ++j){
            int idx = subset[i][j];
            if(idx == -1){
                (poses[subset.size() - i][j]).type = 3;
                continue;
            }
	    (poses[subset.size() - i][j]).type = 1;
	    (poses[subset.size() - i][j]).point = get<2>(candidate[idx]);
        }
    }
    return poses;
}

} }
#endif
