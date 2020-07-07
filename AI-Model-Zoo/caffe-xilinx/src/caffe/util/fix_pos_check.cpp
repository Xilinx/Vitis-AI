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

/*************************************************************************
    > File Name: src/caffe/util/fix_pos_check.cpp
    > Author: Niu Xinjun
    > Mail: xinjun.niu@deephi.tech
    > Created Time: Mon 04 Jun 2018 04:03:15 PM CST
 ************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/fix_pos_check.hpp"
#include "caffe/util/convert_proto.hpp"
#include <vector>

namespace caffe {

  bool checkPosConverge( Net<float> & net ) {
    bool converge = true;
    for ( int i = net.layers().size() - 2; i >= 1; --i ) {
      // bypass the first and last layer. Suppose they are not concat layer.
      auto& curLayer = net.layers()[i];
      auto& bottoms = net.bottom_vecs()[i];
      auto& tops = net.top_vecs()[i];
      if ( curLayer->layer_param().type() == "Concat" ) {
        //printf( "\n........find CONCAT layer.\n" ); fflush(stdout);
        // get minimum fix position
        int minPos = tops[0]->fixed_pos();
        for ( int j = 0; j < bottoms.size(); ++j ) {
          if ( minPos > bottoms[j]->fixed_pos() ) {
            minPos = bottoms[j]->fixed_pos();
            tops[0]->set_fixed_pos( minPos );
            converge = false;
          }
        }
        // set minimum fix position to bottoms
        for ( int j = 0; j < bottoms.size(); ++j ) {
          if ( minPos < bottoms[j]->fixed_pos() ) {
            bottoms[j]->set_fixed_pos( minPos );
            converge = false;
          }
        }
      } else if ( IsFixSkipLayer( curLayer->layer_param() ) ) {
        //  || curLayer->layer_param().type() == "DeephiResize" ) {
        // transfer fix position from top to bottom in Split layer
        //printf( "\n........find FIX SKIP layer.\n" ); fflush(stdout);
        if ( curLayer->layer_param().type() == "Pooling" )
          continue;
        int minPos = tops[0]->fixed_pos();
        for (int i = 0; i < tops.size(); ++i) {
          if ( minPos > tops[i]->fixed_pos() )
            minPos = tops[i]->fixed_pos();
        }
        if ( minPos < 999 ) // all top data fixed position has been set.
          bottoms[0]->set_fixed_pos( minPos );
      }
    }
    return converge;
  }

}
