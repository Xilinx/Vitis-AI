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

#include "ap_int.h"
#include "xf_nms_config.h"
#include "ap_fixed.h"

#define DEBUG 0

#if DEBUG
#include <iostream>
#endif

template<typename T>
T overlap(T x1, T w1, T x2, T w2) {
	//#pragma HLS inline off
	T w1_n = w1 >> 1;
	T w2_n = w2 >> 1;
	//T left = max(x1 - w1_n, x2 - w2_n);
	//T right = min(x1 + w1_n, x2 + w2_n);
	T x11 = x1 - w1_n;
//#pragma HLS BIND_OP variable=x11 op=sub impl=dsp
	T x22 = x2 - w2_n;
//#pragma HLS BIND_OP variable=x22 op=sub impl=dsp
	T x12 = x1 + w1_n;
//#pragma HLS BIND_OP variable=x12 op=add impl=dsp
	T x21 = x2 + w2_n;
//#pragma HLS BIND_OP variable=x21 op=add impl=dsp
	T left = (x11 > x22) ? x11 : x22;
	T right = (x12 < x21)? x12 : x21;
	return (left > right) ? (T)0 : (T)(right - left);
	//return right - left;
}

template<typename T>
bool cal_iou_fixed(T box, T truth, FIX_TYPE nms_th, float& over) {
#pragma HLS inline off
	FIX_TYPE w = overlap(box.x, box.w, truth.x, truth.w);
	FIX_TYPE h = overlap(box.y, box.h, truth.y, truth.h);
	//cout << "\nfix w box : " << (float)w/BW_VAL << "\t" << (float)h/BW_VAL << "\t" << (float)box.x/BW_VAL << "\t"
	//		<< (float)box.w/BW_VAL << "\t" << (float)truth.x/BW_VAL << "\t" <<  (float)truth.w/BW_VAL << "\n";
	//if (w < 0 || h < 0) return true;

	ap_int<32> inter_area = ((ap_int<MUL_BW>)w * (ap_int<MUL_BW>)h);
//#pragma HLS BIND_OP variable=inter_area op=mul impl=dsp
	ap_int<32> box_area = ((ap_int<MUL_BW>)box.w * (ap_int<MUL_BW>)box.h);
//#pragma HLS BIND_OP variable=box_area op=mul impl=dsp
	ap_int<32> truth_area = ((ap_int<MUL_BW>)truth.w * (ap_int<MUL_BW>)truth.h);
//#pragma HLS BIND_OP variable=truth_area op=mul impl=dsp

	ap_int<32> area =  box_area + truth_area - inter_area; // a28

	ap_int<32> lhs = inter_area;
//#pragma HLS BIND_OP variable=lhs op=mul impl=dsp
	ap_int<34> mul_area = (nms_th*area) >> FL_SCORE;
	ap_int<32> rhs = mul_area.range(31, 0);
//#pragma HLS BIND_OP variable=rhs op=mul impl=dsp

	bool ignore = (lhs >= rhs) ? false : true;

	float ai = (float)inter_area/(268435456.0);
	float au = (float)area/(268435456.0);
	over = ai/au;

	//bool ignore =  (over >= 0.596) ? false : true;

	//std::cout<< "ai, au: " << ai << ", " << au << "\n";

#if 0//DEBUG
	std::cout << "\nref:\tAI: " << (float)inter_area/(268435456.0) << "\tAU: " << (float)area/(268435456.0) << "\tAB: " << (float)box_area/(268435456.0)
							<< "\tAT: " << (float)truth_area/(268435456.0) << "\tlhs: " << (float)lhs/(268435456.0)  << "\trhs: " << (float)rhs/(268435456.0) << "\ten: " << ignore << "\n";
	//cout << "\nfix model : " << (float)inter_area/(268435456.0) << "\t" << (float)n_union_area/(BW_VAL) << "\n";
	//cout << "\nfix model : " << (float)inter_area/(268435456.0) << "\t" << (float)union_area/(268435456.0) << "\t 16bit union: " << (float)n_union_area/(4096.0) <<
	//		"\tiou: " << (float)iou/BW_VAL << "\tfxiou: " << iou <<  "\n";
#endif
	return ignore;
}

template<int MAX_ENTRY>
void process(
		s_box box[MAX_ENTRY],
		s_box cur_box,
		ap_uint<MAX_ENTRY> valid,
		bool exist_flag[MAX_ENTRY],
		bool iou_cmp[MAX_ENTRY],
		FIX_TYPE nms_thresh) {
#pragma HLS inline

	bool box_en[MAX_ENTRY];
#pragma HLS ARRAY_PARTITION variable=box_en complete

	Parallel_loop : for(int buf_idx = 0; buf_idx < MAX_ENTRY; buf_idx++) {
#pragma HLS unroll

		float over;
		iou_cmp[buf_idx] = cal_iou_fixed(cur_box, box[buf_idx], nms_thresh, over);

		if(buf_idx == 0) {
			box[buf_idx] = (!valid(buf_idx, buf_idx)) ? cur_box : box[buf_idx];
		} else {
			box[buf_idx] = (valid(buf_idx, buf_idx) xor valid(buf_idx-1, buf_idx-1)) ? cur_box : box[buf_idx];
		}
		box_en[buf_idx] = (!(valid(buf_idx, buf_idx) & exist_flag[buf_idx]) | iou_cmp[buf_idx]);
#if DEBUG
		std::cout << buf_idx << " :- HLS V: " << valid(buf_idx, buf_idx) << ",\t box_en flag: " << box_en[buf_idx] <<
				",\t Exist: " << exist_flag[buf_idx] << ",\tiou_comp: " << iou_cmp[buf_idx] << ",\t over: " << over << "\n";

				//",\t box.x: " << (float)cur_box.x/BW_VAL << ",\t box.x: " << (float)box[buf_idx].x/BW_VAL <<
				//",\t box.y: " << (float)box[buf_idx].y/BW_VAL << "\n";
#endif
	}

	bool enable = true;
	for(int i = 0; i < MAX_ENTRY; i++) {
#pragma HLS PIPELINE
		enable = enable & box_en[i];
	}

	for(int buf_idx = 0; buf_idx < MAX_ENTRY; buf_idx++) {
#pragma HLS unroll
		if(!buf_idx) {
			exist_flag[buf_idx] = (!valid(buf_idx, buf_idx)) ? enable : exist_flag[buf_idx];
		}else{
			exist_flag[buf_idx] = (valid(buf_idx, buf_idx) xor valid(buf_idx-1, buf_idx-1)) ? enable : exist_flag[buf_idx];
		}
		//std::cout << buf_idx << " :- V: " << valid(buf_idx, buf_idx) << ",\t E1 flag: " << box_en << ",\t E: " << exist_flag[buf_idx] << ",\tignore flag: " << iou_cmp[buf_idx] << ",\t box.x: " << box.x << ",\t box.x: " << box[buf_idx].x << ",\t box.y: " << box[buf_idx].y << "\n";
	} // NPC loop end

	return;
}

//# Write Output
template<int MAX_ENTRY, int OUT_WIDTH>
void write_output(
		ap_uint<OUT_WIDTH>* num_out_entries,
		ap_uint<OUT_WIDTH>* outBoxes,
		s_box box[MAX_ENTRY],
		ap_uint<MAX_ENTRY> valid,
		bool exist_flag[MAX_ENTRY]) {
#pragma HLS inline

	ap_uint<OUT_WIDTH> out_box;
	FIX_TYPE box_h = 0;
	ap_uint<16> num_out_boxes = 0;
	//# Run Max entries in a class
	for(int i = 0; i < MAX_ENTRY; i++) {
#pragma HLS PIPELINE
		if(valid(i,i) & exist_flag[i]) {
			out_box.range(15,0) = box[i].x;
			out_box.range(31,16) = box[i].y;
			out_box.range(47,32) = box[i].w;
			out_box.range(63,48) = box[i].h;
			out_box.range(79,64) = box[i].score;

#if ELE_PACK==6
			out_box.range(79,64) = box[i].cls_id;
			out_box.range(95,80) = box[i].score;
#endif
			//# write output
			outBoxes[num_out_boxes++] = out_box;
		}
	}
    *num_out_entries = (ap_uint<OUT_WIDTH>)num_out_boxes;
}

template<int MAX_ENTRY, int IN_WIDTH, int OUT_WIDTH>
void nms(
		hls::stream<ap_int<IN_WIDTH> >& nmsStrm_in,
		ap_uint<OUT_WIDTH>* outBoxes,
		FIX_TYPE nms_thresh,
		int num_entry_perclass)
{
	//# Boolean vector indicating validity of the entries
	//ap_uint<MAX_ENTRY> valid;
	//#pragma HLS ARRAY_PARTITION variable=valid complete dim=1

	ap_uint<MAX_ENTRY> valid = 0;

	//# Boolean vector indicating enablement of an entry for output
	bool exist_flag[MAX_ENTRY];
#pragma HLS ARRAY_PARTITION variable=exist_flag complete dim=1

	bool iou_cmp[MAX_ENTRY];
#pragma HLS ARRAY_PARTITION variable=iou_cmp complete dim=1

	s_box box[MAX_ENTRY], _box;
#pragma HLS ARRAY_PARTITION variable=box complete dim=1

	//# Init buffers
	for(int cl_id = 0; cl_id < MAX_ENTRY; cl_id++)
	{
#pragma HLS unroll
		//valid[cl_id] = false;
		exist_flag[cl_id] = true;
		iou_cmp[cl_id] = false;
        /*box[cl_id].x = 0;
        box[cl_id].y = 0;
        box[cl_id].w = 0;
        box[cl_id].h = 0;*/
	}

	//# loop over the boxes
	for(int r = 0; r < num_entry_perclass; r++) {
#pragma HLS LOOP_TRIPCOUNT min=50 max=50
#pragma HLS PIPELINE II=1

		//ap_uint<IN_WIDTH> r_box_in = inBoxes[r];
		ap_uint<IN_WIDTH> r_box_in = nmsStrm_in.read();

		//read new entry
		_box.x  = (ap_int<BW>)r_box_in.range(15, 0);
		_box.y  = (ap_int<BW>)r_box_in.range(31,16);
		_box.w = (ap_int<BW>)r_box_in.range(47,32);
		_box.h = (ap_int<BW>)r_box_in.range(63,48);
		_box.score = (ap_int<BW>)r_box_in.range(79,64);
		//box.score = (ap_int<BW>)r_box_in.range(95,80);

#if DEBUG
		std::cout << "entry : " << r << "\n";
#endif
		process<MAX_ENTRY>(box, _box, valid, exist_flag, iou_cmp, nms_thresh);

		valid(MAX_ENTRY-1, 1) = valid.range(MAX_ENTRY-2, 0);
		valid(0,0) = 1;
	}
	write_output<MAX_ENTRY, OUT_WIDTH>(&outBoxes[0], &outBoxes[1], box, valid, exist_flag);
}

template<int MAX_ENTRY, int IN_WIDTH, int OUT_WIDTH>
void nmsTop(
		hls::stream<ap_int<IN_WIDTH> >& nmsStrm_in,
		ap_uint<OUT_WIDTH>* outBoxes,
		int num_entry_perclass,
		short int nms_fxin)
{
	FIX_TYPE nms_thresh = (FIX_TYPE)nms_fxin;
	nms<MAX_ENTRY, IN_WIDTH, OUT_WIDTH>(nmsStrm_in, outBoxes, nms_thresh, num_entry_perclass);

	return;
}
