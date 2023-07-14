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
#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>

namespace vitis { namespace ai {

class PointPillarsPre
{
  public:
    PointPillarsPre(std::vector<int8_t*>& in_addr1,  int in_scale1, int in_width1, int in_height1, int in_channel1,
                    std::vector<int8_t*>& out_addr1, float out_scale1, int out_width1, int out_height1, int out_channel1,
                    std::vector<int8_t*>& in_addr2,  int in_scale2, int in_width2, int in_height2, int in_channel2,
                    int batchnum, int& realbatchnum
    );
    ~PointPillarsPre();

    void process_net0(const float* points, int len, int batchidx);
    void process_net1(int batchidx);
    void process_net1_cleanmem();
    void process_net1_thread( int start, int len, int batchidx);


    inline bool judge_op_same(int canvas_index, int threadidx);
    void process_net0_thread(const float* points , int idx, int start, int len, int&, int batchidx);

    std::vector<std::shared_ptr<preout_dict>>   pre_dict_;

  private:

    std::array<int, 3> voxelmap_shape_;
    V1I coor_to_voxelidx;

    std::vector<int8_t*> in_addr1_;

    int in_scale1_;
    int in_height1_;
    std::vector<int8_t*> out_addr1_;
    float out_scale1_;
    std::vector<int8_t*> in_addr2_;
    int in_scale2_, in_width2_, in_height2_, in_channel2_;
    int cfg_max_number_of_points_per_voxel;
    V1F cfg_voxel_size;
    int cfg_max_number_of_voxels;
    bool bDirect;

    std::vector<std::thread> vth0;
    std::mutex mtx;
    int PRE_MT_NUM = 2;
    int XLNX_POINTPILLARS_MIDDLE_MT = 2;
    int canvas_index_arr[8];

    int batchnum = 0;
    int& realbatchnum;

    std::array<float, 3> scale_pclen;
    std::array<float, 3> scale_pcstartlen;
    std::array<float, 3> point_range;
};


}}


