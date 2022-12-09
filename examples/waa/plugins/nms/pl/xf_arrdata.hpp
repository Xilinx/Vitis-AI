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
#pragma once

#include "xf_arrdata_config.h"
//#include "ap_fixed.h"

#define AR_DEBUG 0

static ap_int<LUT_BW> lut_y_x_val[512] = {-1638, -1626, -1613, -1600, -1587, -1574, -1562, -1549, -1536, -1523, -1510, -1498, -1485, -1472, -1459, -1446, -1434, -1421, -1408, -1395, -1382, -1370, -1357, -1344, -1331, -1318, -1306, -1293, -1280, -1267, -1254, -1242, -1229, -1216, -1203, -1190, -1178, -1165, -1152, -1139, -1126, -1114, -1101, -1088, -1075, -1062, -1050, -1037, -1024, -1011, -998, -986, -973, -960, -947, -934, -922, -909, -896, -883, -870, -858, -845, -832, -819, -806, -794, -781, -768, -755, -742, -730, -717, -704, -691, -678, -666, -653, -640, -627, -614, -602, -589, -576, -563, -550, -538, -525, -512, -499, -486, -474, -461, -448, -435, -422, -410, -397, -384, -371, -358, -346, -333, -320, -307, -294, -282, -269, -256, -243, -230, -218, -205, -192, -179, -166, -154, -141, -128, -115, -102, -90, -77, -64, -51, -38, -26, -13, 0, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128, 141, 154, 166, 179, 192, 205, 218, 230, 243, 256, 269, 282, 294, 307, 320, 333, 346, 358, 371, 384, 397, 410, 422, 435, 448, 461, 474, 486, 499, 512, 525, 538, 550, 563, 576, 589, 602, 614, 627, 640, 653, 666, 678, 691, 704, 717, 730, 742, 755, 768, 781, 794, 806, 819, 832, 845, 858, 870, 883, 896, 909, 922, 934, 947, 960, 973, 986, 998, 1011, 1024, 1037, 1050, 1062, 1075, 1088, 1101, 1114, 1126, 1139, 1152, 1165, 1178, 1190, 1203, 1216, 1229, 1242, 1254, 1267, 1280, 1293, 1306, 1318, 1331, 1344, 1357, 1370, 1382, 1395, 1408, 1421, 1434, 1446, 1459, 1472, 1485, 1498, 1510, 1523, 1536, 1549, 1562, 1574, 1587, 1600, 1613, 1626, -1638, -1626, -1613, -1600, -1587, -1574, -1562, -1549, -1536, -1523, -1510, -1498, -1485, -1472, -1459, -1446, -1434, -1421, -1408, -1395, -1382, -1370, -1357, -1344, -1331, -1318, -1306, -1293, -1280, -1267, -1254, -1242, -1229, -1216, -1203, -1190, -1178, -1165, -1152, -1139, -1126, -1114, -1101, -1088, -1075, -1062, -1050, -1037, -1024, -1011, -998, -986, -973, -960, -947, -934, -922, -909, -896, -883, -870, -858, -845, -832, -819, -806, -794, -781, -768, -755, -742, -730, -717, -704, -691, -678, -666, -653, -640, -627, -614, -602, -589, -576, -563, -550, -538, -525, -512, -499, -486, -474, -461, -448, -435, -422, -410, -397, -384, -371, -358, -346, -333, -320, -307, -294, -282, -269, -256, -243, -230, -218, -205, -192, -179, -166, -154, -141, -128, -115, -102, -90, -77, -64, -51, -38, -26, -13, 0, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128, 141, 154, 166, 179, 192, 205, 218, 230, 243, 256, 269, 282, 294, 307, 320, 333, 346, 358, 371, 384, 397, 410, 422, 435, 448, 461, 474, 486, 499, 512, 525, 538, 550, 563, 576, 589, 602, 614, 627, 640, 653, 666, 678, 691, 704, 717, 730, 742, 755, 768, 781, 794, 806, 819, 832, 845, 858, 870, 883, 896, 909, 922, 934, 947, 960, 973, 986, 998, 1011, 1024, 1037, 1050, 1062, 1075, 1088, 1101, 1114, 1126, 1139, 1152, 1165, 1178, 1190, 1203, 1216, 1229, 1242, 1254, 1267, 1280, 1293, 1306, 1318, 1331, 1344, 1357, 1370, 1382, 1395, 1408, 1421, 1434, 1446, 1459, 1472, 1485, 1498, 1510, 1523, 1536, 1549, 1562, 1574, 1587, 1600, 1613, 1626};
static ap_int<LUT_BW> lut_h_w_val[512] = {42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 61, 62, 64, 65, 67, 69, 71, 72, 74, 76, 78, 80, 82, 84, 86, 88, 91, 93, 95, 98, 100, 103, 105, 108, 111, 113, 116, 119, 122, 125, 129, 132, 135, 139, 142, 146, 149, 153, 157, 161, 165, 169, 174, 178, 182, 187, 192, 197, 202, 207, 212, 217, 223, 228, 234, 240, 246, 253, 259, 265, 272, 279, 286, 293, 301, 308, 316, 324, 332, 341, 349, 358, 367, 377, 386, 396, 406, 416, 427, 438, 449, 460, 472, 484, 496, 509, 521, 535, 548, 562, 576, 591, 606, 621, 637, 653, 669, 686, 704, 722, 740, 759, 778, 797, 818, 838, 860, 881, 904, 927, 950, 974, 999, 1024, 1050, 1077, 1104, 1132, 1160, 1190, 1220, 1251, 1282, 1315, 1348, 1382, 1417, 1453, 1490, 1528, 1566, 1606, 1647, 1688, 1731, 1775, 1820, 1866, 1913, 1962, 2011, 2062, 2114, 2168, 2223, 2279, 2337, 2396, 2456, 2519, 2582, 2648, 2715, 2784, 2854, 2926, 3000, 3076, 3154, 3234, 3316, 3400, 3486, 3574, 3665, 3757, 3852, 3950, 4050, 4153, 4258, 4365, 4476, 4589, 4705, 4825, 4947, 5072, 5200, 5332, 5467, 5605, 5747, 5893, 6042, 6195, 6352, 6512, 6677, 6846, 7020, 7197, 7380, 7566, 7758, 7954, 8156, 8362, 8574, 8791, 9013, 9242, 9476, 9715, 9961, 10214, 10472, 10737, 11009, 11288, 11573, 11866, 12167, 12475, 12791, 13114, 13446, 13787, 14136, 14494, 14861, 15237, 15623, 16018, 16424, 16839, 17266, 17703, 18151, 18610, 19081, 19564, 20060, 20568, 21088, 21622, 22169, 22731, 23306, 23896, 24501, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 61, 62, 64, 65, 67, 69, 71, 72, 74, 76, 78, 80, 82, 84, 86, 88, 91, 93, 95, 98, 100, 103, 105, 108, 111, 113, 116, 119, 122, 125, 129, 132, 135, 139, 142, 146, 149, 153, 157, 161, 165, 169, 174, 178, 182, 187, 192, 197, 202, 207, 212, 217, 223, 228, 234, 240, 246, 253, 259, 265, 272, 279, 286, 293, 301, 308, 316, 324, 332, 341, 349, 358, 367, 377, 386, 396, 406, 416, 427, 438, 449, 460, 472, 484, 496, 509, 521, 535, 548, 562, 576, 591, 606, 621, 637, 653, 669, 686, 704, 722, 740, 759, 778, 797, 818, 838, 860, 881, 904, 927, 950, 974, 999, 1024, 1050, 1077, 1104, 1132, 1160, 1190, 1220, 1251, 1282, 1315, 1348, 1382, 1417, 1453, 1490, 1528, 1566, 1606, 1647, 1688, 1731, 1775, 1820, 1866, 1913, 1962, 2011, 2062, 2114, 2168, 2223, 2279, 2337, 2396, 2456, 2519, 2582, 2648, 2715, 2784, 2854, 2926, 3000, 3076, 3154, 3234, 3316, 3400, 3486, 3574, 3665, 3757, 3852, 3950, 4050, 4153, 4258, 4365, 4476, 4589, 4705, 4825, 4947, 5072, 5200, 5332, 5467, 5605, 5747, 5893, 6042, 6195, 6352, 6512, 6677, 6846, 7020, 7197, 7380, 7566, 7758, 7954, 8156, 8362, 8574, 8791, 9013, 9242, 9476, 9715, 9961, 10214, 10472, 10737, 11009, 11288, 11573, 11866, 12167, 12475, 12791, 13114, 13446, 13787, 14136, 14494, 14861, 15237, 15623, 16018, 16424, 16839, 17266, 17703, 18151, 18610, 19081, 19564, 20060, 20568, 21088, 21622, 22169, 22731, 23306, 23896, 24501};

#if (GL_TEST==0)
template < int PR_PTR_WIDTH, int IN_WIDTH, int OUT_WIDH, int LUT_PTR_WIDTH, int MAX_ENTRY, int BUFFERBITS, int BUFFERDEPTH>
void arradataTop(
		ap_uint<PR_PTR_WIDTH>* priors,
		ap_uint<IN_WIDTH>* inBoxes1,
		ap_uint<IN_WIDTH>* inBoxes2,
		ap_uint<IN_WIDTH>* inBoxes3,
		ap_uint<IN_WIDTH>* inBoxes4,
		hls::stream<ap_int<OUT_WIDH> >& strm_out,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		short int sort_size,
		int cl_id)
{
#pragma HLS dependence variable=lut_y_x_val inter RAW false
#pragma HLS resource variable=lut_y_x_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_y_x_val ram_2p

#pragma HLS dependence variable=lut_h_w_val inter RAW false
#pragma HLS resource variable=lut_h_w_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_h_w_val ram_2p

	/*
	ap_uint<PR_PTR_WIDTH> lut_val;
	for(short int id = 0; id <= 255; id++) {
#pragma HLS PIPELINE II=1
		lut_val = lut_ptr[id];

		lut_y_x_val[id] = (ap_int<16>)lut_val(15,0);//lut_ptr[id];
		lut_y_x_val[id + 256] = (ap_int<16>)lut_val(31,16);//lut_ptr[id + 256];
		lut_h_w_val[id] = (ap_int<16>)lut_val(47,32);//lut_ptr[id + 512];
		lut_h_w_val[id +256 ] = (ap_int<16>)lut_val(63,48);//lut_ptr[id + 768];
	}
	 */
	ap_uint<IN_WIDTH> bbox_tmp;
	ap_uint<OUT_WIDH> out;
	ap_uint<32> bbox;
	ap_uint<PR_PTR_WIDTH> prior_bbox;

	short int cur_srtidx = 0;
	int box_index = 0;

#if AR_DEBUG
	std::cout << "hw sort size : " << sort_size << "\n";
#endif

	//std::cout << "**************hw class: " << cl_id << ", size: " << sort_size << " ***************\n";
	for(int i = 0; i < sort_size; i++ )
	{
#pragma HLS LOOP_TRIPCOUNT min=50 max=50
#pragma HLS PIPELINE II=1

		int div_idx	= i/NCPC;
		int pos_rem	= i - div_idx*NCPC;
		ap_uint<BUFFERBITS*NUM_COMP> var = TopKBuffer[cl_id][div_idx];
		int dis = pos_rem*BUFFERBITS;

		int8_t score = (int8_t)var(dis+DATA_BITS-1, dis);
		short cur_srtidx = (short)var(dis+BUFFERBITS-1, DATA_BITS+dis);

		prior_bbox = priors[cur_srtidx];

		ap_int<16> y_center_a = (ap_int<16>)prior_bbox.range(15, 0);  //2.6
		ap_int<16> x_center_a = (ap_int<16>)prior_bbox.range(31, 16);
		ap_int<16> ha = (ap_int<16>)prior_bbox.range(47, 32);
		ap_int<16> wa = (ap_int<16>)prior_bbox.range(63, 48);

		//# Read input box
		//bbox = inBoxes1[cur_srtidx];

		/*ap_int<8> bbox_y = (ap_int<8>)bbox.range(7, 0);
		ap_int<8> bbox_x = (ap_int<8>)bbox.range(15, 8);
		ap_int<8> bbox_h = (ap_int<8>)bbox.range(23, 16);
		ap_int<8> bbox_w = (ap_int<8>)bbox.range(31, 24);*/

		box_index = cur_srtidx*4;
		ap_int<8> bbox_y = (ap_int<8>)inBoxes1[box_index];
		ap_int<8> bbox_x = (ap_int<8>)inBoxes2[box_index+1];
		ap_int<8> bbox_h = (ap_int<8>)inBoxes3[box_index+2];
		ap_int<8> bbox_w = (ap_int<8>)inBoxes4[box_index+3];


		//std::cout << "hw sort index, inbox_idx, inbox_dis: " << cur_srtidx << ", " << inbox_idx << ", " << inbox_dis << "\n";
		//std::cout << "in y,x,h,w: " << bbox_y << ", " << bbox_x << ", " << bbox_h << ", " << bbox_w << "\n";

		ap_int<16> ty = lut_y_x_val[bbox_y + 128] ; // 6.10
		ap_int<16> tx = lut_y_x_val[bbox_x + 384]; // 6.10

		//# Compute h & w
		ap_int<16> h = (lut_h_w_val[bbox_h + 128] * ha) >> 8;
		ap_int<16> w = (lut_h_w_val[bbox_w + 384] * wa) >> 8;

		ap_int<16> xc1 = (tx * wa) >> 10; // 6.10 * 4.12 = 10.22 => 10.12 => 4.12
		ap_int<16> yc1 = (ty * ha) >> 10;

#if AR_DEBUG
		std::cout << "yc, xc, ha, wa: " << y_center_a/64.0 << ", " << x_center_a/64.0 << ", " << ha/64.0 << ", " << wa/64.0 << "\n";
		std::cout << "HW ty: " << ty/256.0 << ", tx: " << tx/256.0 << ", exp(th): " << lut_h_w_val[bbox_h + 128]/256.0 << ", ha: " << ha/64.0 << ", h: " << h/16384.0 << ", w: " << w/16384.0 << "\n";
#endif

		//# Compute xcenter and ycenter
		ap_int<16> ycenter1 = yc1 + y_center_a; // 8.8 * 2.6 + 2.6 * 8.8 = 10.14
		ap_int<16> xcenter1 = xc1 + x_center_a;

		ap_int<16> xcenter = xcenter1 << 2;
		ap_int<16> ycenter = ycenter1 << 2;

		out.range(15,0) = xcenter;
		out.range(31,16) = ycenter;
		out.range(47,32) = w ;
		out.range(63,48) = h ;
		out.range(79,64) = score ;

		//std::cout << "out xc, yc, w, h: " << (float)xcenter/16384.0 << ", " << (float)ycenter/16384.0 << ", " << (float)w/16384.0 << ", " << (float)h/16384.0 << "\n";

		//# Write out data
		strm_out.write(out);
	}
}
#else
template < int PR_PTR_WIDTH, int IN_WIDTH, int OUT_WIDH, int LUT_PTR_WIDTH, int MAX_ENTRY, int BUFFERBITS, int BUFFERDEPTH>
void arradataTop(
		ap_uint<PR_PTR_WIDTH>* priors,
		ap_uint<IN_WIDTH>* inBoxes,
		ap_uint<64> *gmemout,
		ap_uint<LUT_PTR_WIDTH>* lut_ptr,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		short int sort_size,
		int cl_id)
{
	static ap_int<LUT_BW> lut_y_x_val[512];
#pragma HLS dependence variable=lut_y_x_val inter RAW false
#pragma HLS resource variable=lut_y_x_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_y_x_val ram_2p

	static ap_int<LUT_BW> lut_h_w_val[512];
#pragma HLS dependence variable=lut_h_w_val inter RAW false
#pragma HLS resource variable=lut_h_w_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_h_w_val ram_2p


	for(short int id = 0; id <= 255; id++) {
#pragma HLS PIPELINE II=1
		lut_y_x_val[id] = (ap_int<16>)lut_ptr[id];
		lut_y_x_val[id + 256] = (ap_int<16>)lut_ptr[id + 256];
		lut_h_w_val[id] = (ap_int<16>)lut_ptr[id + 512];
		lut_h_w_val[id +256 ] = (ap_int<16>)lut_ptr[id + 768];
	}

	ap_uint<IN_WIDTH> bbox_tmp;
	ap_uint<OUT_WIDH> out;
	ap_uint<32> bbox;
	ap_uint<PR_PTR_WIDTH> prior_bbox;

	short int cur_srtidx = 0;

#if AR_DEBUG
	std::cout << "hw sort size : " << sort_size << "\n";
#endif

	//std::cout << "**************hw class: " << cl_id << ", size: " << sort_size << " ***************\n";
	for(int i = 0; i < sort_size; i++ )
	{
#pragma HLS LOOP_TRIPCOUNT min=50 max=50
#pragma HLS PIPELINE II=1

		int srt_idx_pos	= i%NCPC;
		int srt_idx_idx	= i/NCPC;
		ap_uint<BUFFERBITS*NUM_COMP> var = TopKBuffer[cl_id][srt_idx_idx];
		int dis = srt_idx_pos*BUFFERBITS;

		int8_t score = (int8_t)var(dis+DATA_BITS-1, dis);
		short cur_srtidx = (short)var(dis+BUFFERBITS-1, DATA_BITS+dis);

		prior_bbox = priors[cur_srtidx];

		/*ap_int<8> y_center_a = (ap_int<8>)prior_bbox.range(7, 0);  //2.6
		ap_int<8> x_center_a = (ap_int<8>)prior_bbox.range(15, 8);
		ap_int<8> ha = (ap_int<8>)prior_bbox.range(23, 16);
		ap_int<8> wa = (ap_int<8>)prior_bbox.range(31, 24);*/

		ap_int<16> y_center_a = (ap_int<16>)prior_bbox.range(15, 0);  //2.6
		ap_int<16> x_center_a = (ap_int<16>)prior_bbox.range(31, 16);
		ap_int<16> ha = (ap_int<16>)prior_bbox.range(47, 32);
		ap_int<16> wa = (ap_int<16>)prior_bbox.range(63, 48);

		int inbox_pos	= cur_srtidx%4;
		int inbox_idx	= cur_srtidx/4;
		int inbox_dis = inbox_pos*32;

		//std::cout << "bufidx, dis: " << cur_srtidx << ", " << inbox_dis << "\n";

		//# Read input box
		bbox_tmp = inBoxes[inbox_idx];
		bbox = bbox_tmp(inbox_dis+31, inbox_dis);

		ap_int<8> bbox_y = (ap_int<8>)bbox.range(7, 0);
		ap_int<8> bbox_x = (ap_int<8>)bbox.range(15, 8);
		ap_int<8> bbox_h = (ap_int<8>)bbox.range(23, 16);
		ap_int<8> bbox_w = (ap_int<8>)bbox.range(31, 24);

		//std::cout << "hw sort index, inbox_idx, inbox_dis: " << cur_srtidx << ", " << inbox_idx << ", " << inbox_dis << "\n";
		//std::cout << "in y,x,h,w: " << bbox_y << ", " << bbox_x << ", " << bbox_h << ", " << bbox_w << "\n";

		ap_int<16> ty = lut_y_x_val[bbox_y + 128] ; // 6.10
		ap_int<16> tx = lut_y_x_val[bbox_x + 384]; // 6.10

		//# Compute h & w
		/*ap_int<24> h1 = (lut_h_w_val[bbox_h + 128] * ha) >> 8; // 6.10 * 4.12 = 10.22 => 10.14
		ap_int<16> h = h1(15, 0); // 2.14

		ap_int<24> w1 = (lut_h_w_val[bbox_w + 384] * wa) >> 8;
		ap_int<16> w = w1(15, 0);*/

		/*ap_int<22> xc11 = (tx * wa) >> 10; // 6.10 * 4.12 = 10.22 => 10.12
		ap_int<16> xc1 = xc11(15, 0); // 4.12

		ap_int<22> yc11 = (ty * ha) >> 10;
		ap_int<16> yc1 = yc11(15, 0); // 4.12*/

		ap_int<16> h = (lut_h_w_val[bbox_h + 128] * ha) >> 8;
		ap_int<16> w = (lut_h_w_val[bbox_w + 384] * wa) >> 8;

		ap_int<16> xc1 = (tx * wa) >> 10; // 6.10 * 4.12 = 10.22 => 10.12 => 4.12
		ap_int<16> yc1 = (ty * ha) >> 10;

#if AR_DEBUG
		std::cout << "yc, xc, ha, wa: " << y_center_a/64.0 << ", " << x_center_a/64.0 << ", " << ha/64.0 << ", " << wa/64.0 << "\n";
		std::cout << "HW ty: " << ty/256.0 << ", tx: " << tx/256.0 << ", exp(th): " << lut_h_w_val[bbox_h + 128]/256.0 << ", ha: " << ha/64.0 << ", h: " << h/16384.0 << ", w: " << w/16384.0 << "\n";
#endif

		//# Compute xcenter and ycenter
		ap_int<16> ycenter1 = yc1 + y_center_a; // 8.8 * 2.6 + 2.6 * 8.8 = 10.14
		ap_int<16> xcenter1 = xc1 + x_center_a;

		ap_int<16> xcenter = xcenter1 << 2;
		ap_int<16> ycenter = ycenter1 << 2;

		out.range(15,0) = xcenter;
		out.range(31,16) = ycenter;
		out.range(47,32) = w ;
		out.range(63,48) = h ;

		std::cout << "hw out xc, yc, w, h: " << (float)xcenter/16384.0 << ", " << (float)ycenter/16384.0 << ", " << (float)w/16384.0 << ", " << (float)h/16384.0 << "\n";

		//# Write out data
		gmemout[i] = out;
		//strm_out[i].writeout);
	}
}
#endif

#if 0
template < int PR_PTR_WIDTH, int IN_WIDTH, int OUT_WIDH, int LUT_PTR_WIDTH, int MAX_ENTRY, int BUFFERBITS, int BUFFERDEPTH>
void arradataTop(
		ap_uint<PR_PTR_WIDTH>* priors,
		ap_uint<IN_WIDTH>* inBoxes,
		ap_uint<64> *gmemout,
		ap_uint<LUT_PTR_WIDTH>* lut_ptr,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		short int sort_size,
		int cl_id)
{
	static ap_int<LUT_BW> lut_y_x_val[512];
#pragma HLS dependence variable=lut_y_x_val inter RAW false
#pragma HLS resource variable=lut_y_x_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_y_x_val ram_2p

	static ap_int<LUT_BW> lut_h_w_val[512];
#pragma HLS dependence variable=lut_h_w_val inter RAW false
#pragma HLS resource variable=lut_h_w_val core=RAM_2P_BRAM
	//#pragma HLS BIND_STORAGE variable=lut_h_w_val ram_2p


	for(short int id = 0; id <= 255; id++) {
#pragma HLS PIPELINE II=1
		lut_y_x_val[id] = (ap_int<16>)lut_ptr[id];
		lut_y_x_val[id + 256] = (ap_int<16>)lut_ptr[id + 256];
		lut_h_w_val[id] = (ap_int<16>)lut_ptr[id + 512];
		lut_h_w_val[id +256 ] = (ap_int<16>)lut_ptr[id + 768];
	}

	ap_uint<IN_WIDTH> bbox_tmp;
	ap_uint<OUT_WIDH> out;
	ap_uint<32> bbox;
	ap_uint<PR_PTR_WIDTH> prior_bbox;

	short int cur_srtidx = 0;

#if AR_DEBUG
	std::cout << "hw sort size : " << sort_size << "\n";
#endif

	std::cout << "**************hw class: " << cl_id << ", size: " << sort_size << " ***************\n";
	for(int i = 0; i < sort_size; i++ )
	{
#pragma HLS LOOP_TRIPCOUNT min=50 max=50
#pragma HLS PIPELINE II=1

		int srt_idx_pos	= i%NCPC;
		int srt_idx_idx	= i/NCPC;
		ap_uint<BUFFERBITS*NUM_COMP> var = TopKBuffer[cl_id][srt_idx_idx];
		int dis = srt_idx_pos*BUFFERBITS;

		int8_t score = (int8_t)var(dis+DATA_BITS-1, dis);
		short cur_srtidx = (short)var(dis+BUFFERBITS-1, DATA_BITS+dis);

		prior_bbox = priors[cur_srtidx];

		ap_int<8> y_center_a = (ap_int<8>)prior_bbox.range(7, 0);  //2.6
		ap_int<8> x_center_a = (ap_int<8>)prior_bbox.range(15, 8);
		ap_int<8> ha = (ap_int<8>)prior_bbox.range(23, 16);
		ap_int<8> wa = (ap_int<8>)prior_bbox.range(31, 24);

		int inbox_pos	= cur_srtidx%4;
		int inbox_idx	= cur_srtidx/4;
		int inbox_dis = inbox_pos*32;

		//std::cout << "bufidx, dis: " << cur_srtidx << ", " << inbox_dis << "\n";

		//# Read input box
		bbox_tmp = inBoxes[inbox_idx];
		bbox = bbox_tmp(inbox_dis+31, inbox_dis);

		ap_int<8> bbox_y = (ap_int<8>)bbox.range(7, 0);
		ap_int<8> bbox_x = (ap_int<8>)bbox.range(15, 8);
		ap_int<8> bbox_h = (ap_int<8>)bbox.range(23, 16);
		ap_int<8> bbox_w = (ap_int<8>)bbox.range(31, 24);

		std::cout << "hw sort index, inbox_idx, inbox_dis: " << cur_srtidx << ", " << inbox_idx << ", " << inbox_dis << "\n";
		std::cout << "in y,x,h,w: " << bbox_y << ", " << bbox_x << ", " << bbox_h << ", " << bbox_w << "\n";
		//std::cout << "128bit: " << bbox_tmp()

		ap_int<BW> ty = lut_y_x_val[bbox_y + 128] ; // 8.8
		ap_int<BW> tx = lut_y_x_val[bbox_x + 384]; //8.8

		//# Compute h & w
		ap_int<HI_BW> h = lut_h_w_val[bbox_h + 128] * ha ; // 8.8 * 2.6 = 10.14
		ap_int<HI_BW> w = lut_h_w_val[bbox_w + 384] * wa ; // 10.14

#if AR_DEBUG
		std::cout << "yc, xc, ha, wa: " << y_center_a/64.0 << ", " << x_center_a/64.0 << ", " << ha/64.0 << ", " << wa/64.0 << "\n";
		std::cout << "HW ty: " << ty/256.0 << ", tx: " << tx/256.0 << ", exp(th): " << lut_h_w_val[bbox_h + 128]/256.0 << ", ha: " << ha/64.0 << ", h: " << h/16384.0 << ", w: " << w/16384.0 << "\n";
#endif

		//# Compute xcenter and ycenter
		ap_int<16> ycenter = (ty * ha) + ((ap_int<16>)y_center_a << 8); // 8.8 * 2.6 + 2.6 * 8.8 = 10.14
		ap_int<16> xcenter = (tx * wa) + ((ap_int<16>)x_center_a << 8);

		out.range(15,0) = xcenter ;
		out.range(31,16) = ycenter ;
		out.range(47,32) = w ;
		out.range(63,48) = h ;

		std::cout << "out xc, yc, w, h: " << (float)xcenter/16384.0 << ", " << (float)ycenter/16384.0 << ", " << (float)w/16384.0 << ", " << (float)h/16384.0 << "\n";

		//# Write out data
		gmemout[i] = out;
		//strm_out[i].writeout);
	}
}
#endif
