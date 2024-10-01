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

#include <cmath>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"

using namespace std;

// env var control if only decode the box which score greater than confidence threshold
DEF_ENV_PARAM(FILTER_BY_SCORE, "1")
// env var control the nms confidence threshold: 0.001 for accuracy
DEF_ENV_PARAM_2(CONF_THRESH, "0.3", float)

namespace {

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }
// softmax: the input data is not sequence, it jumps the step for each value
static void softmax_v(float* input, float* output, size_t cls, size_t step) {
    double sum = 0.0;
    vector<double> tmp(cls);
    for (auto i = 0u; i < cls; ++i) {
      tmp[i] = std::exp((double)input[i*step]);
      sum += tmp[i];
    }
    for (unsigned int i = 0u; i < cls; ++i) {
      output[i] = tmp[i] / sum;
    }
}

static void myprintv(float* v, int len)
{
 std::cout <<"dump:\n";  for(int i=0; i<len; i++) { std::cout << v[i] << " "; } std::cout <<"\n";
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}{
      bfilter = ENV_PARAM(FILTER_BY_SCORE);
      conf_thresh = ENV_PARAM(CONF_THRESH );
  }

  int calculate(vart::simple_tensor_buffer_t<void> output, std::vector<vart::simple_tensor_buffer_t<void>> input) {
    CHECK_EQ(input.size(), 3);
    //name YOLOv2D19__YOLOv2D19__YOLOv2D19_Conv2d_pred__prediction_sink_tranpose_0_fix 84500: 1x125x13x13 x4(for float)
    //name YOLOv2D19__YOLOv2D19_grid_cell                                              1352:  1x169x1x2 x4
    //name YOLOv2D19__YOLOv2D19_all_anchor_wh                                          6760:  1x169x5x2 x4

    std::sort( inlayeridx.begin(),
               inlayeridx.end(),
               [&input]( int a, int b){ return input[a].mem_size < input[b].mem_size; } ); // 1 2 0
    CHECK_EQ( input[inlayeridx[0]].mem_size*5, input[inlayeridx[1]].mem_size);

    // inlayeridx[0] : grid_cell       1x169x1x2
    // inlayeridx[1] : all_anchor_wh   1x169x5x2
    // inlayeridx[2] : prediction      1x125x13x13
    input_shape = input[inlayeridx[1]].tensor->get_shape(); // 1 169 5 2
    CHECK_EQ(input_shape.size(), 4);
    CHECK_EQ(input_shape[3], 2);
    n_anchors = input_shape[2];

    input_shape = input[inlayeridx[2]].tensor->get_shape(); // 1 125 13 13 // nchw
    CHECK_EQ(input_shape.size(), 4);
    nclasses = (input_shape[1]-n_anchors- 4*n_anchors)/n_anchors ; // (125-5-4*5)/5==20;
    std::vector<float> sfmxout(input_shape[2]*input_shape[3]*n_anchors*nclasses);
    float* sfmxin = (float*)input[inlayeridx[2] ].data;
    float* outlayer = (float*)output.data;
    int retnum = input_shape[2]*input_shape[3]*n_anchors;  // 13x13x5
    int cls_inds = 0;
    memset((void*)outlayer, 0, input_shape[2]*input_shape[3]*n_anchors*4*3*sizeof(float));

    float conf_pred=0.0;

    if(0) myprintv((float*)input[ inlayeridx[0]].data, input[ inlayeridx[0]].mem_size/sizeof(float) );
    if(0) myprintv((float*)input[ inlayeridx[1]].data, input[ inlayeridx[1]].mem_size/sizeof(float) );
    if(0) myprintv((float*)input[ inlayeridx[2]].data, input[ inlayeridx[2]].mem_size/sizeof(float) );

    for(int i=0; i<input_shape[2]; i++) {   // h 13
      for(int j=0; j<input_shape[3]; j++) { // w 13
        for(int k=0; k<n_anchors; k++) {    //   5
           int base = i*input_shape[3]*n_anchors + j*n_anchors+k;   // 13x13x5
           // int src_pos  = i*input_shape[2]*input_shape[3] +j*input_shape[3] + n_anchors +k*nclasses; //old
           int src_pos  = (n_anchors+ k*nclasses)*input_shape[2]*input_shape[3] + i*input_shape[3]+ j;
           int dst_pos  = base*nclasses;
           softmax_v( sfmxin+src_pos, sfmxout.data()+dst_pos, nclasses, input_shape[2]*input_shape[3]);

           int conf_pos = k*input_shape[2]*input_shape[3] +i*input_shape[3] +j;
           conf_pred = sigmoid( sfmxin[conf_pos] );
           for(int kk=0; kk<nclasses; kk++) {
              sfmxout[ dst_pos+kk ] *= conf_pred;
           }

           cls_inds = std::max_element( sfmxout.data()+dst_pos, sfmxout.data()+dst_pos+nclasses) - (sfmxout.data()+dst_pos);
           if ( bfilter && sfmxout[dst_pos+cls_inds] < conf_thresh ) {
              continue;
           }
           auto box = get_decode_boxes(sfmxin,
                                       (float*)input[ inlayeridx[0]].data,
                                       (float*)input[ inlayeridx[1]].data,
                                       base);
           for(int kk=0; kk<4; kk++) {
             outlayer[(base+retnum*0)*4+kk] = box[kk];
             outlayer[(base+retnum*1)*4+kk] = sfmxout[dst_pos+cls_inds ];
             outlayer[(base+retnum*2)*4+kk] = cls_inds;
           }
        } // end of for k
      } // end of for j
    } // end of for i
    return 0;
  }

private:
  std::vector<float> get_decode_boxes(float* in_data, float* grid_cell, float* anchor_wh, int in_pos) {
     std::vector<float>  ret(4, 0.0);
     int i=in_pos/n_anchors,  j=in_pos%n_anchors;         //  h13_w13 5
     int i0=i/input_shape[3], j0=i%input_shape[3];  //  h13 w13
     int step = input_shape[2]*input_shape[3];
     int pos = (n_anchors+n_anchors*nclasses + j*4 )*input_shape[2]*input_shape[3] + i0*input_shape[3]+j0;

     float f0 = stride*(sigmoid( in_data[ pos + step*0 ]) + grid_cell[i*2 + 0] );
     float f1 = stride*(sigmoid( in_data[ pos + step*1 ]) + grid_cell[i*2 + 1] );
     float f2 = stride*(std::exp(in_data[ pos + step*2 ]) * anchor_wh[i*n_anchors*2 + j*2 + 0] );
     float f3 = stride*(std::exp(in_data[ pos + step*3 ]) * anchor_wh[i*n_anchors*2 + j*2 + 1] );

     ret[0] = std::clamp((f0 - f2/2.0)/input_size, 0.0, 1.0 );
     ret[1] = std::clamp((f1 - f3/2.0)/input_size, 0.0, 1.0 );
     ret[2] = std::clamp((f0 + f2/2.0)/input_size, 0.0, 1.0 );
     ret[3] = std::clamp((f1 + f3/2.0)/input_size, 0.0, 1.0 );
     return ret;
  }

private:
  const int stride = 32;
  const int input_size = 416;

  std::vector<int> inlayeridx{0, 1 ,2 };
  int n_anchors = 5;
  std::vector<std::int32_t> input_shape;
  int nclasses = 20;
  float conf_thresh=0.3; // for normal
  // float conf_thresh=0.001; // for accuracy
  bool bfilter = true;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
