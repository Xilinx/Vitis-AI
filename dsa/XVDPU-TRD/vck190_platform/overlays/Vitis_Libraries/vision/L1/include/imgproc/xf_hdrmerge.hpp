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

#ifndef _XF_HDRMERGE_HPP_
#define _XF_HDRMERGE_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

/**Utility macros and functions**/

template <typename T>
T xf_satcast_hdr(int in_val){};

template <>
inline ap_uint<8> xf_satcast_hdr<ap_uint<8> >(int v) {
    return (v > 255 ? 255 : v);
};
template <>
inline ap_uint<10> xf_satcast_hdr<ap_uint<10> >(int v) {
    return (v > 1023 ? 1023 : v);
};
template <>
inline ap_uint<12> xf_satcast_hdr<ap_uint<12> >(int v) {
    return (v > 4095 ? 4095 : v);
};
template <>
inline ap_uint<16> xf_satcast_hdr<ap_uint<16> >(int v) {
    return (v > 65535 ? 65535 : v);
};

namespace xf {
namespace cv {

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int NO_EXPS, int W_SIZE>
void Hdrmerge_bayer(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat1,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat2,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                    short wr_hls[NO_EXPS * NPC * W_SIZE]) {
// clang-format off
	#pragma HLS ARRAY_PARTITION variable=wr_hls dim=1 block factor=NO_EXPS*NPC
// clang-format on	
	unsigned short width = _src_mat1.cols >> XF_BITSHIFT(NPC);
	unsigned short height = _src_mat1.rows;

	int rdindex = 0;
	int wrindex = 0;
	const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);
	XF_TNAME(SRC_T, NPC) val_src1,val_src2;
    XF_TNAME(DST_T, NPC) val_dst;


	//FILE *fp1 = fopen("imagevals_hls.txt","w");
	for(int i = 0;i< height;i++){
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
    // clang-format on		
		for(int j=0;j< width;j++){
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
            // clang-format on
            val_src1 = (_src_mat1.read(rdindex));
            val_src2 = (_src_mat2.read(rdindex++));

            for (int p = 0, n = 0; p < XF_NPIXPERCYCLE(NPC); p++, n = NO_EXPS * p) {
// clang-format off
#pragma HLS unroll
                // clang-format on				
			XF_CTUNAME(SRC_T, NPC) val1 = val_src1.range(p * STEP + STEP - 1, p * STEP);
			XF_CTUNAME(SRC_T, NPC) val2 = val_src2.range(p * STEP + STEP - 1, p * STEP);
			
			int index1 = (n*W_SIZE)+val1;
			int index2 = ((n+1)*W_SIZE)+val2;
			
			short final_w1 =  (short)(wr_hls[index1]);
			short final_w2 =  (short)(wr_hls[index2]);
		
			ap_fixed<STEP+STEP*2,STEP+STEP> val_1 = (ap_fixed<STEP+STEP*2,STEP+STEP>)((float)(final_w1 * val1)/16384);
			ap_fixed<STEP+STEP*2,STEP+STEP> val_2 = (ap_fixed<STEP+STEP*2,STEP+STEP>)((float)(final_w2 * val2)/16384);
			
			ap_fixed<STEP+STEP*2,STEP+STEP> sum_wei = (ap_fixed<STEP+STEP*2,STEP+STEP>)((float)(final_w1+final_w2)/16384);

			int final_val = (int)((val_1+val_2) / sum_wei);
			
			XF_CTUNAME(SRC_T, NPC) out_val = xf_satcast_hdr<XF_CTUNAME(SRC_T, NPC)>(final_val);
			
			val_dst.range(p * STEP + STEP - 1, p * STEP) = out_val;
			
			}
			_dst_mat.write(wrindex++,(val_dst));
		}
		//fprintf(fp1,"\n");
	}
	//fclose(fp1);
}
} // namespace cv
} // namespace xf

#endif
