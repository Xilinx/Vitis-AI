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
#include "hls_stream.h"
#include "xf_blas.hpp"

template <unsigned int t_ParEntries, typename t_DataType>
class Activation {
   public:
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries> t_WideType;
    typedef typename t_WideType::t_TypeInt t_TypeInt;

    static t_DataType sigmoid(t_DataType x) {
        t_DataType l_exp = hls::exp<t_DataType>(-x);
        return 1.0f / (1.0f + l_exp);
    }

    static void sigmoid(unsigned int p_n, hls::stream<t_TypeInt>& p_x, hls::stream<t_TypeInt>& p_y) {
#ifndef __SYNTHESIS__
        assert(p_n % t_ParEntries == 0);
#endif
        int l_numParEntries = p_n / t_ParEntries;
        for (int n = 0; n < l_numParEntries; ++n) {
#pragma HLS PIPELINE
            t_WideType l_valX;
            t_WideType l_valY;
            l_valX = p_x.read();
            for (int i = 0; i < t_ParEntries; ++i) l_valY[i] = sigmoid(-l_valX[i]);
            p_y.write(l_valY);
        }
    }
};
