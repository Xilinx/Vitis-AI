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

#ifndef _XF_SORT_HPP_
#define _XF_SORT_HPP_

//#include "common/xf_common.hpp"
#include <assert.h>
#include "ap_int.h"

#define SPMAX 127//2147483647

namespace xf {
namespace cv {

#if 1
template <int NUM_CLASS,
int NCPC,
int DATABITS,
int IN_PTR_WIDTH,
int SORTORDER,
int INDEXBITS, int TOPK>
void Read_input(ap_uint<IN_PTR_WIDTH>* src,
		hls::stream<ap_int<DATABITS> > inStrm[NUM_CLASS],
		int n,
		hls::stream<ap_uint<INDEXBITS> > &Strm_class_size
)
{
#pragma HLS ARRAY_PARTITION variable=inStrm complete dim=1
#pragma HLS inline off

	const float nms_threshold=-3.38919;
	const char nms_threshold_8bit = -4;

	enum{
		NUM_CLASS_FLOOR=(NUM_CLASS/NCPC)*NCPC
	};

	int loop_bound = (n*NUM_CLASS + NCPC - 1)/NCPC;
	int class_cnt=0;

	ap_uint<3> m_sel[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=m_sel complete dim=1
	bool class_wr_en[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=class_wr_en complete dim=1
	short class_size[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=class_size complete dim=1
	for(int cl_id=0; cl_id<NUM_CLASS; cl_id++)
	{
#pragma HLS unroll
		m_sel[cl_id] = cl_id;
		if(cl_id<NCPC)
			class_wr_en[cl_id] = 1;
		else
			class_wr_en[cl_id] = 0;
		class_size[cl_id]=0;
	}

	//fprintf(stderr,"\nread loop=%d",(int)loop_bound);
	for(int index=0;index<loop_bound;index++){
#pragma HLS LOOP_TRIPCOUNT min=2000*90/8 max=2000*90/8
#pragma HLS PIPELINE
		ap_uint<IN_PTR_WIDTH> data_npc = src[index];
		ap_int<DATABITS> data_int8[NCPC];
#pragma HLS ARRAY_PARTITION variable=data_int8 complete dim=1
		for(int id=0, bit=0; id<NCPC; id++, bit+=8){
#pragma HLS unroll
			data_int8[id] = data_npc.range(bit+7, bit);
		}
		//fprintf(stderr,"\n********** LOOP= %d*******",index);
		for(int cl_id=0; cl_id<NUM_CLASS; cl_id++)
		{
#pragma HLS unroll
			//fprintf(stderr,"\n<%d>class index:en  =  %d:%d",cl_id, (int)m_sel[cl_id],class_wr_en[cl_id]);

			if(class_wr_en[cl_id]==1 && !((index==loop_bound-1)&&(cl_id<NCPC)) ){
				//if(index==loop_bound-1)
				//   fprintf(stderr,"\n<ptr:%d> classnum:%d  msel=%d",index, cl_id, (int)m_sel[cl_id]);
				ap_int<DATABITS> data_wr = data_int8[m_sel[cl_id]];

				if(data_wr > nms_threshold_8bit)
				{
					class_size[cl_id]++;
				}

				inStrm[cl_id].write(data_wr);
			}
		}

		ap_uint<3> m_sel_tmp[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=m_sel_tmp complete dim=1
		bool class_wr_en_tmp[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=class_wr_en_tmp complete dim=1

		for(int cl_id=0; cl_id<NUM_CLASS; cl_id++)
		{
#pragma HLS unroll

			if(cl_id>=NCPC)
			{
				m_sel_tmp[cl_id] = m_sel[cl_id-NCPC];
				class_wr_en_tmp[cl_id] = class_wr_en[cl_id-NCPC];
			}
			else
			{
				m_sel_tmp[cl_id] = m_sel[cl_id+NUM_CLASS-NCPC];
				class_wr_en_tmp[cl_id] = class_wr_en[cl_id+NUM_CLASS-NCPC];
			}
		}

		for(int cl_id=0; cl_id<NUM_CLASS; cl_id++)
		{
#pragma HLS unroll

			m_sel[cl_id] = m_sel_tmp[cl_id];
			class_wr_en[cl_id] = class_wr_en_tmp[cl_id];
		}


		//  data_int8[NCPC]
		//  m_sel[NUM_CLASS], class_wr[NUM_CLASS]
		// if (class_wr[i])
		//   stream[i] = data_int8[m_sel[i]]
		//		{
		//			ap_uint<8> classid;
		//			if((class_cnt+id) > (NUM_CLASS-1))
		//				classid = (class_cnt+id) - NUM_CLASS;
		//			else
		//				classid = (class_cnt+id);
		//			if(index<(loop_bound-1) || class_cnt+id < NUM_CLASS)
		//				inStrm[classid].write(data_int8);
		//		}//end loop id
		//		class_cnt+=NCPC;
		//		if(class_cnt>=NUM_CLASS)
		//			class_cnt-=NUM_CLASS;

	}//end loop index

	for(int cl=0; cl<NUM_CLASS;cl++)
	{
#pragma HLS pipeline
		short num_valid_entry_tmp;
		if(class_size[cl]>TOPK)
			num_valid_entry_tmp = TOPK;
		else
			num_valid_entry_tmp = class_size[cl];

		Strm_class_size.write(num_valid_entry_tmp);
	}

}
#endif

template <int NUM_CLASS,
int TOPK,
int NUM_COMP,
int OUT_PTR_WIDTH_INDEX,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int INDEXBITS_OUT,
int BUFFERBITS,
int DEPTH_SIZE_BUFF>
void Write_outputBuffer_size(hls::stream<ap_uint<INDEXBITS> > &dstStrm,
		ap_uint<INDEXBITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF])
{

	short tmp_buffer[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=tmp_buffer complete dim=1
	ap_uint<OUT_PTR_WIDTH_INDEX> merge_data;
	short wr_index=0;
	char tmp_buff_cnt=0;

	int wr_cnt=0;

	for(short i=0;i<NUM_CLASS;i++){
#pragma HLS PIPELINE II=1
		int size_tmp  = dstStrm.read();
		int size;
		if(size_tmp>TOPK)
			size=TOPK;
		else
			size=size_tmp;
		tmp_buffer[tmp_buff_cnt] = size;

		for(int nc=0, bit=0; nc<NUM_COMP; nc++,bit+=INDEXBITS_OUT)
		{
#pragma HLS unroll
			merge_data.range(bit+15, bit) = tmp_buffer[nc];
		}

		if( (tmp_buff_cnt==NUM_COMP-1) | (i==NUM_CLASS-1))
		{
			//dst_size[wr_index++] = merge_data;
			outSize_of_class[wr_index++] = merge_data;
			tmp_buff_cnt=0;
			wr_cnt++;
		}
		else
		{
			tmp_buff_cnt++;
		}
		//dst_size[i] = (short)size;
	}
	//	std::cout << "write size cnt " << wr_cnt << std::endl;
	int stop=1;
}

#if 0
template <int NUM_CLASS,
int TOPK,
int NUM_COMP,
int OUT_PTR_WIDTH,
int OUT_PTR_WIDTH_INDEX,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int BUFFERBITS,
int OUTSTREAMBITS,
int OUTSTREAMBITS_SCORE,
int OUTSTREAMBITS_INDEX>
void Write_output(
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		//hls::stream<ap_uint<OUTSTREAMBITS_SCORE> > dstStrm_score[NUM_CLASS],
		//hls::stream<ap_uint<OUTSTREAMBITS_INDEX> > dstStrm_index[NUM_CLASS],
		ap_uint<OUT_PTR_WIDTH>* dst_score,
		//ap_uint<OUT_PTR_WIDTH_INDEX>* dst_index)
		ap_uint<OUT_PTR_WIDTH>* dst_index)//,
//hls::stream<ap_uint<INDEXBITS> > dstStrm_size[NUM_CLASS],
//ap_uint<OUT_PTR_WIDTH>* dst_size)
{
#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1
#pragma HLS inline off
	/*
	short index=0;
	LW1:for(ap_uint<8> d2=0;d2<BUFFERDEPTH;d2++){
#pragma HLS PIPELINE II=1
		ap_uint<OUTSTREAMBITS_SCORE> score_packdata;//adatapack_score;
		ap_uint<OUTSTREAMBITS_INDEX> index_packdata;//datapack_index;
		LW2:for(int d1=0, bit=0, bit_score=0, bit_index=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS, bit_score+=DATABITS, bit_index+=INDEXBITS){
#pragma HLS unroll
			ap_uint<BUFFERBITS> datatmp = TopKBuffer[d1][d2];
			ap_int<DATABITS> datascore  = (ap_int<DATABITS>)datatmp.range(DATABITS-1,0);
			ap_uint<INDEXBITS> dataidx = (ap_uint<INDEXBITS>)datatmp.range(BUFFERBITS-1,DATABITS);
		        score_packdata.range(bit_score+DATABITS-1, bit_score) = datascore;
			index_packdata.range(bit_index+INDEXBITS-1, bit_index) = dataidx;
		}
		dstStrm_score.write(score_packdata);
		dstStrm_index.write(index_packdata);
	}
	 */
	int wr_cnt=0;

	short index1=0;
	LW0_S:for(short class_idx=0;class_idx<NUM_CLASS;class_idx++){
		LW1_S:for(ap_uint<8> d2=0;d2<BUFFERDEPTH;d2++){
#pragma HLS PIPELINE II=1
			ap_uint<OUTSTREAMBITS_SCORE> score_packdata;//adatapack_score;
			ap_uint<BUFFERBITS*NUM_COMP> topk_read = TopKBuffer[class_idx][d2];
			LW2_S:for(int d1=0, bit=0, bit_score=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS, bit_score+=DATABITS){
#pragma HLS unroll
				ap_uint<BUFFERBITS> datatmp = topk_read.range(bit+BUFFERBITS-1,bit);
				ap_int<DATABITS> datascore  = (ap_int<DATABITS>)datatmp.range(DATABITS-1,0);
				score_packdata.range(bit_score+DATABITS-1, bit_score) = datascore;
			}
			dst_score[index1++] = score_packdata;
			wr_cnt++;
		}
	}

	short index2=0;
	LW0_I:for(short class_idx=0;class_idx<NUM_CLASS;class_idx++){
		LW1_I:for(ap_uint<8> d2=0;d2<BUFFERDEPTH*2;d2++){
#pragma HLS PIPELINE II=1
			ap_uint<OUT_PTR_WIDTH*2> index_packdata;//datapack_index;
			ap_uint<BUFFERBITS*NUM_COMP> topk_read = TopKBuffer[class_idx][d2/2];
			LW2_I:for(int d1=0, bit=0, bit_index=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS, bit_index+=16){
#pragma HLS unroll
				ap_uint<BUFFERBITS> datatmp =  topk_read.range(bit+BUFFERBITS-1,bit);//TopKBuffer[class_idx][d1][d2/2];
				short dataidx  = (short)datatmp.range(BUFFERBITS-1,DATABITS);
				index_packdata.range(bit_index+16-1, bit_index) = dataidx;
			}
			if(d2[0]==0)
				dst_index[index2++] = index_packdata.range(OUT_PTR_WIDTH-1,0);
			else
				dst_index[index2++] = index_packdata.range(2*OUT_PTR_WIDTH-1,OUT_PTR_WIDTH);
			wr_cnt++;
		}
	}

	//	std::cout << "write score index cnt " << wr_cnt << std::endl;

	/*
	int index=0;
	L1:for(short class_idx=0;class_idx<NUM_CLASS;class_idx++){
		for(short idx=0;idx<BUFFERDEPTH;idx++){
#pragma HLS PIPELINE II=1
			ap_uint<OUT_PTR_WIDTH> score_packdata;
			for(int index_comp=0, bit_index=0, bit_outstrm=0; index_comp < NUM_COMP; index_comp++, bit_index+=16, bit_outstrm+=INDEXBITS)
			{
#pragma HLS unroll
				ap_uint<INDEXBITS> dataidx  = datapack.range(bit_outstrm-1+INDEXBITS,bit_outstrm) ;
				index_packdata.range(bit_index+16-1, bit_index) = (short)dataidx;
			}
			dst_score[index] = datapack;
			index++;
		}
	}

	index=0;
	L2:for(short class_idx=0;class_idx<NUM_CLASS;class_idx++){
		for(short idx=0;idx<BUFFERDEPTH;idx++){
#pragma HLS PIPELINE II=2
			ap_uint<2*OUT_PTR_WIDTH> index_packdata;
			ap_uint<OUTSTREAMBITS_INDEX> datapack = dstStrm_index[class_idx].read();
			for(int index_comp=0, bit_index=0, bit_outstrm=0; index_comp < NUM_COMP; index_comp++, bit_index+=16, bit_outstrm+=INDEXBITS)
			{
#pragma HLS unroll
				ap_uint<INDEXBITS> dataidx  = datapack.range(bit_outstrm-1+INDEXBITS,bit_outstrm) ;
				index_packdata.range(bit_index+16-1, bit_index) = (short)dataidx;
			}
			dst_index[2*index] = index_packdata.range(OUT_PTR_WIDTH_INDEX/2-1,0);
			dst_index[2*index+1] = index_packdata.range(OUT_PTR_WIDTH_INDEX-1,OUT_PTR_WIDTH_INDEX/2);
			index++;
		}
	}
	 */
	/*
	short tmp_buffer[NUM_COMP/2];
#pragma HLS ARRAY_PARTITION variable=tmp_buffer complete dim=1
	ap_uint<OUT_PTR_WIDTH> merge_data;
	short wr_index=0;
	char tmp_buff_cnt=0;
	L3:for(short i=0;i<NUM_CLASS;i++){
#pragma HLS PIPELINE II=1
		int size_tmp  = dstStrm_size[i].read();
		int size;
		if(size_tmp>TOPK)
			size=TOPK;
		else
			size=size_tmp;
		tmp_buffer[tmp_buff_cnt] = size;

		for(int nc=0, bit=0; nc<NUM_COMP/2; nc++,bit+=16)
		{
#pragma HLS unroll
		  merge_data.range(bit+15, bit) = tmp_buffer[nc]; 
		}

		if( (tmp_buff_cnt==NUM_COMP/2-1) | (i==NUM_CLASS-1))
		{
		   dst_size[wr_index++] = merge_data;
		   tmp_buff_cnt=0;
		}
		else
		{
		   tmp_buff_cnt++;
		}
		//dst_size[i] = (short)size;
	}
	 */

}
#endif

//# Disabled for NMS integration
#if 0//new write
template <int NUM_CLASS,
int TOPK,
int NUM_COMP,
int OUT_PTR_WIDTH,
int OUT_PTR_WIDTH_INDEX,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int INDEXBITS_OUT,
int BUFFERBITS,
int OUTSTREAMBITS,
int OUTSTREAMBITS_SCORE,
int OUTSTREAMBITS_INDEX,
int DEPTH_SIZE_BUFF>
void Write_output(
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		ap_uint<INDEXBITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF],
		ap_uint<OUT_PTR_WIDTH>* dst,
		short loop_bound
)
{
#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1
#pragma HLS inline off

	int wr_cnt=0;
	short index1=0;
	ap_uint<8> buffer_index=0;
	short class_idx=0;
	short class_inc_cnt=0;
	short out_size_idx=0;

	LW0_SI:for(short wr_idx=0;wr_idx<loop_bound;wr_idx++){
#pragma HLS DEPENDENCE variable=TopKBuffer inter false
#pragma HLS DEPENDENCE variable=TopKBuffer intra false
		//	LW0_SI:for(short wr_idx=0;wr_idx<NUM_CLASS*((BUFFERDEPTH/2)+BUFFERDEPTH);wr_idx++){
#pragma HLS pipeline II=1
		ap_uint<DATABITS*NUM_COMP> score_packdata1;
		ap_uint<DATABITS*NUM_COMP> score_packdata2;
		ap_uint<OUT_PTR_WIDTH> index_packdata;
		ap_uint<BUFFERBITS*NUM_COMP> topk_read1 = TopKBuffer[class_idx][buffer_index];
		ap_uint<BUFFERBITS*NUM_COMP> topk_read2 = TopKBuffer[class_idx][buffer_index+1];
		LW2_S:for(int d1=0, bit=0, bit_score=0, bit_index=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS, bit_score+=DATABITS, bit_index+=INDEXBITS_OUT){
#pragma HLS unroll
			ap_uint<BUFFERBITS> datatmp1 = topk_read1.range(bit+BUFFERBITS-1,bit);
			ap_int<DATABITS> datascore1  = (ap_int<DATABITS>)datatmp1.range(DATABITS-1,0);
			score_packdata1.range(bit_score+DATABITS-1, bit_score) = datascore1;

			short dataidx  = (short)datatmp1.range(BUFFERBITS-1,DATABITS);
			//std::cout << "HLS dataidx: " << dataidx << "\n";
			index_packdata.range(bit_index+INDEXBITS_OUT-1, bit_index) = dataidx;

			ap_uint<BUFFERBITS> datatmp2 = topk_read2.range(bit+BUFFERBITS-1,bit);
			ap_int<DATABITS> datascore2  = (ap_int<DATABITS>)datatmp2.range(DATABITS-1,0);
			score_packdata2.range(bit_score+DATABITS-1, bit_score) = datascore2;
		}
		ap_uint<OUT_PTR_WIDTH> outdata_score;
		outdata_score.range((OUT_PTR_WIDTH/2)-1,0) = score_packdata1;
		outdata_score.range((OUT_PTR_WIDTH)-1,(OUT_PTR_WIDTH/2)) = score_packdata2;
		ap_uint<OUT_PTR_WIDTH> outdata_index = index_packdata;

		ap_uint<OUT_PTR_WIDTH> outdata_scoreindex;
		ap_uint<8> inc_buffer_index;
		short class_inc_cnt_max;
		if(wr_idx < NUM_CLASS*(BUFFERDEPTH/2))//write score to DDR
		{
			outdata_scoreindex = outdata_score;
			inc_buffer_index=2;
			class_inc_cnt_max=(BUFFERDEPTH/2);
		}
		else
		{
			outdata_scoreindex = outdata_index;
			inc_buffer_index=1;
			class_inc_cnt_max=BUFFERDEPTH;
		}

		ap_uint<OUT_PTR_WIDTH> outdata;
		if(wr_idx<(NUM_CLASS*((BUFFERDEPTH/2)+BUFFERDEPTH)))
			outdata = outdata_scoreindex;
		else
			outdata = outSize_of_class[out_size_idx++];

		if(wr_idx == NUM_CLASS*(BUFFERDEPTH/2)-1)
			class_idx=0;
		else if(class_inc_cnt==class_inc_cnt_max-1)
			class_idx++;

		if(class_inc_cnt==class_inc_cnt_max-1){
			class_inc_cnt=0;
			//class_idx++;
			buffer_index=0;
		}
		else
		{
			class_inc_cnt++;
			buffer_index+=inc_buffer_index;
		}

		dst[wr_idx] = outdata;
	}


}
#endif

//5th jun
#if 1
template <int TOPK,
int SORTORDER,
int NUM_COMP,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int BUFFERBITS,
int OUT_PTR_WIDTH,
int OUTSTREAMBITS,
int OUTSTREAMBITS_SCORE,
int OUTSTREAMBITS_INDEX>
void Sort_1class(hls::stream<ap_int<DATABITS> >& srcStrm,

		int n,
		int loop_bound_process,
		//		hls::stream<ap_uint<OUTSTREAMBITS_SCORE> > &dstStrm_score,
		//		hls::stream<ap_uint<OUTSTREAMBITS_INDEX> > &dstStrm_index,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[BUFFERDEPTH])
{
#pragma HLS inline off
#pragma HLS resource variable=TopKBuffer core=RAM_2P_LUTRAM

	const float nms_threshold=-3.38919;
	const char nms_threshold_8bit = -4;//(char)nms_threshold;
	//fprintf(stderr,"\n--- nms th= %d", (int)nms_threshold_8bit);

	bool comp_prev_reg;
	ap_uint<BUFFERBITS> E_prev_reg;
	ap_int<DATABITS> score;

	ap_uint<BUFFERBITS> score_idx;
	ap_uint<12> in_idx=0;
	ap_uint<8> topk_idx=0;
	//fprintf(stderr,"\nProcess loop=%d",(int)loop_bound_process);
	L1:for(short idx_in_topk=0;idx_in_topk<loop_bound_process;idx_in_topk++){
#pragma HLS LOOP_TRIPCOUNT min=2000*11 max=2000*11
#pragma HLS DEPENDENCE variable=TopKBuffer inter false
#pragma HLS PIPELINE II=1

		topk_idx = idx_in_topk%BUFFERDEPTH;
		in_idx = idx_in_topk/BUFFERDEPTH;
		bool first_entry=(in_idx==0);
		/*if(topk_idx==0){
			ap_int<DATABITS> score = srcStrm.read();
			score_idx.range(DATABITS-1,0) = score;
			score_idx.range(BUFFERBITS-1,DATABITS) = in_idx;
			//			idx_in++;
		}
		*/

		if(topk_idx==0){
			ap_int<DATABITS> score_tmp = srcStrm.read();
			if(score_tmp > nms_threshold_8bit)
			{
				score = score_tmp;
				//num_valid_entry++;
			}
			else
				if(SORTORDER==0)
					score = -(SPMAX+1);
				else
					score = SPMAX;
			score_idx.range(DATABITS-1,0) = score;
			score_idx.range(BUFFERBITS-1,DATABITS) = in_idx;
			//			idx_in++;
		}
		//		if(topk_idx==BUFFERDEPTH-1){
		//
		//			in_idx++;
		//		}

		ap_uint<BUFFERBITS> E[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=E complete dim=0
		bool comp[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=comp complete dim=0
		ap_uint<BUFFERBITS*NUM_COMP> topk_read = TopKBuffer[topk_idx];
		for(int idx_comp=0, bit=0; idx_comp< NUM_COMP; idx_comp++, bit+=BUFFERBITS)
		{
#pragma HLS unroll
			ap_uint<BUFFERBITS> topk_data = topk_read.range(bit+BUFFERBITS-1,bit);
			ap_int<DATABITS> Eval;
			if(first_entry)//in_idx==0)
			{
				if(SORTORDER==0)
					Eval = -(SPMAX+1);
				else
					Eval = SPMAX;
			}
			else
				Eval = topk_data.range(DATABITS-1,0);

			E[idx_comp].range(DATABITS-1,0) = Eval;
			E[idx_comp].range(BUFFERBITS-1,DATABITS) = topk_data.range(BUFFERBITS-1,DATABITS);

			if(SORTORDER==0)
				comp[idx_comp] = score > Eval;
			else
				comp[idx_comp] = score < Eval;

		}


		ap_uint<BUFFERBITS*NUM_COMP> topk_write;// = TopKBuffer[topk_idx];
		for(int idx_comp=0, bit=0; idx_comp< NUM_COMP; idx_comp++, bit+=BUFFERBITS)
		{
#pragma HLS unroll
			bool comp_prev_val;
			ap_uint<BUFFERBITS> E_prev_val;
			if(idx_comp==0)
			{
				if(topk_idx==0){
					comp_prev_val = 0;
					E_prev_val = E[idx_comp];
				}else{
					comp_prev_val = comp_prev_reg;
					E_prev_val = E_prev_reg;
				}
			}else{
				comp_prev_val = comp[idx_comp-1];
				E_prev_val = E[idx_comp-1];
			}

			topk_write.range(bit+BUFFERBITS-1,bit)= (comp[idx_comp] xor comp_prev_val) ? score_idx : ((comp[idx_comp]) ? E_prev_val : E[idx_comp]);

		}
		TopKBuffer[topk_idx]=topk_write;

		E_prev_reg = E[NUM_COMP-1];
		comp_prev_reg = comp[NUM_COMP-1];
		//		if(topk_idx==BUFFERDEPTH-1)
		//		{
		//			topk_idx=0;
		//		}
		//		else
		//			topk_idx++;
	}//idx_in_topk

	/*	short index=0;
	LW1:for(ap_uint<8> d2=0;d2<BUFFERDEPTH;d2++){
#pragma HLS PIPELINE II=1
		ap_uint<BUFFERBITS*NCPC> datapack;
		LW2:for(int d1=0, bit=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS){
#pragma HLS unroll
			ap_uint<BUFFERBITS> datatmp = TopKBuffer[d1][d2];
			datapack.range(bit+BUFFERBITS-1,bit) = datatmp;
		}
		dstStrm.write(datapack);
	}
	 */

	/*
	short index=0;
	LW1:for(ap_uint<8> d2=0;d2<BUFFERDEPTH;d2++){
#pragma HLS PIPELINE II=1
		ap_uint<OUTSTREAMBITS_SCORE> score_packdata;//adatapack_score;
		ap_uint<OUTSTREAMBITS_INDEX> index_packdata;//datapack_index;
		LW2:for(int d1=0, bit=0, bit_score=0, bit_index=0;d1<NUM_COMP;d1++, bit+=BUFFERBITS, bit_score+=DATABITS, bit_index+=INDEXBITS){
#pragma HLS unroll
			ap_uint<BUFFERBITS> datatmp = TopKBuffer[d1][d2];
			ap_int<DATABITS> datascore  = (ap_int<DATABITS>)datatmp.range(DATABITS-1,0);
			ap_uint<INDEXBITS> dataidx = (ap_uint<INDEXBITS>)datatmp.range(BUFFERBITS-1,DATABITS);
		        score_packdata.range(bit_score+DATABITS-1, bit_score) = datascore;
			index_packdata.range(bit_index+INDEXBITS-1, bit_index) = dataidx;
		}
		dstStrm_score.write(score_packdata);
		dstStrm_index.write(index_packdata);
	}
	 */

}
#endif

//4th jun
#if 0

template <int TOPK,
int SORTORDER,
int NUM_COMP,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int BUFFERBITS,
int OUT_PTR_WIDTH>
void Sort_1class(hls::stream<ap_int<DATABITS> >& srcStrm,
		//		ap_uint<BUFFERBITS> TopKBuffer[NUM_COMP][BUFFERDEPTH],
		int n,
		ap_uint<24> loop_bound_process,
		hls::stream<ap_uint<BUFFERBITS> >& dstStrm)
{

#pragma HLS inline off

	ap_uint<BUFFERBITS> TopKBuffer[NUM_COMP][BUFFERDEPTH];
#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1

	bool comp_prev_reg;
	ap_uint<BUFFERBITS> E_prev_reg;
	ap_int<DATABITS> score;
#if 1

	ap_uint<BUFFERBITS> score_idx;
	int idx_in=0;
	int idx_topk=0;
	L1:for(int idx_in_topk=0;idx_in_topk<loop_bound_process;idx_in_topk++){
#pragma HLS LOOP_TRIPCOUNT min=2000*11 max=2000*11
#pragma HLS PIPELINE II=1

		idx_topk = idx_in_topk%BUFFERDEPTH;
		idx_in = idx_in_topk/BUFFERDEPTH;

		if(idx_topk==0){
			score = srcStrm.read();
			score_idx.range(DATABITS-1,0) = score;
			score_idx.range(BUFFERBITS-1,DATABITS) = idx_in;

		}


		int s = score;

		ap_uint<BUFFERBITS> E[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=E complete dim=0
		bool comp[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=comp complete dim=0
		ap_uint<BUFFERBITS> Eprev[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=Eprev complete dim=0
		for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
		{
#pragma HLS unroll
			ap_uint<BUFFERBITS> topk_data = TopKBuffer[idx_comp][idx_topk];
			ap_int<DATABITS> Eval;
			if(idx_in==0)
			{
				if(SORTORDER==0)
					Eval = -(SPMAX+1);
				else
					Eval = SPMAX;
			}
			else
				Eval = topk_data.range(DATABITS-1,0);

			int a = idx_comp;
			int eeevaal = Eval;

			E[idx_comp].range(DATABITS-1,0) = Eval;
			E[idx_comp].range(BUFFERBITS-1,DATABITS) = topk_data.range(BUFFERBITS-1,DATABITS);

			if(SORTORDER==0)
				comp[idx_comp] = score >= Eval;
			else
				comp[idx_comp] = score <= Eval;

			bool c = comp[idx_comp];
			int stop=1;
		}
		for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
		{
#pragma HLS unroll

			if(idx_comp==0)
				if(idx_topk==0)
					Eprev[idx_comp] = E[idx_comp];
				else
					Eprev[idx_comp] = E_prev_reg;
			else
				Eprev[idx_comp] = E[idx_comp-1];

			int eeevaalPrev_reg = (int)((ap_int<8>)E_prev_reg.range(DATABITS-1,0));
			int eeevaalPrev = (int)((ap_int<8>)E[idx_comp].range(DATABITS-1,0));

			int IDeeevaalPrev_reg = (int)((ap_int<8>)E_prev_reg.range(BUFFERBITS-1,DATABITS));
			int IDeeevaalPrev = (int)((ap_int<8>)E[idx_comp].range(BUFFERBITS-1,DATABITS));

			int stop=1;
		}

		bool comp_prev_val;
		if(idx_topk==0)
			comp_prev_val = 0;
		else
			comp_prev_val = comp_prev_reg;

		//fprintf(stderr,"\n");
		for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
		{
#pragma HLS unroll

			//				int c_prev = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(DATABITS-1,0));

			if(idx_comp!=0)
				TopKBuffer[idx_comp][idx_topk] = (comp[idx_comp] xor comp[idx_comp-1]) ? score_idx : ((comp[idx_comp]) ? Eprev[idx_comp] : E[idx_comp]);
			else
				TopKBuffer[idx_comp][idx_topk] = (comp[idx_comp] xor comp_prev_val) ? score_idx : ((comp[idx_comp]) ? Eprev[idx_comp] : E[idx_comp]);

			//								int a = idx_comp;
			//								int b = idx_topk;
			//								int c = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(DATABITS-1,0));
			//								int d = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(BUFFERBITS-1,DATABITS));
			//								fprintf(stderr,"\tTopKBuffer[%d][%d] = %d,%d",idx_comp,idx_topk,c,d);
		}

		E_prev_reg = E[NUM_COMP-1];
		comp_prev_reg = comp[NUM_COMP-1];

		int c3 = (int)E_prev_reg.range(DATABITS-1,0);
		int stop=0;

		//	TopKBuffer[idx_topk] = CompMux<TOPK,SORTORDER>(score, &E_prev_reg, &C_prev_reg, TopKBuffer[idx_topk], idx_topk);
		//		if(idx_topk==BUFFERDEPTH-1)
		//		{
		//			idx_topk=0;
		//			idx_in++;
		//		}
		//		else
		//			idx_topk++;
	}//idx_in_topk

#else
	ap_uint<BUFFERBITS> score_idx;
	L1:for(int idx_in=0;idx_in<n;idx_in++){
#pragma HLS loop_flatten off
#pragma HLS LOOP_TRIPCOUNT min=2000 max=2000
		L2:for(int idx_topk=0;idx_topk<(BUFFERDEPTH);idx_topk++){
#pragma HLS loop_flatten off
#pragma HLS PIPELINE II=1

			if(idx_topk==0){
				score = srcStrm.read();
				score_idx.range(DATABITS-1,0) = score;
				score_idx.range(BUFFERBITS-1,DATABITS) = idx_in;

			}


			int s = score;

			ap_uint<BUFFERBITS> E[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=E complete dim=0
			bool comp[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=comp complete dim=0
			ap_uint<BUFFERBITS> Eprev[NUM_COMP];
#pragma HLS ARRAY_PARTITION variable=Eprev complete dim=0
			for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
			{
#pragma HLS unroll
				ap_uint<BUFFERBITS> topk_data = TopKBuffer[idx_comp][idx_topk];
				ap_int<DATABITS> Eval;
				if(idx_in==0)
				{
					if(SORTORDER==0)
						Eval = -(SPMAX+1);
					else
						Eval = SPMAX;
				}
				else
					Eval = topk_data.range(DATABITS-1,0);

				int a = idx_comp;
				int eeevaal = Eval;

				E[idx_comp].range(DATABITS-1,0) = Eval;
				E[idx_comp].range(BUFFERBITS-1,DATABITS) = topk_data.range(BUFFERBITS-1,DATABITS);

				if(SORTORDER==0)
					comp[idx_comp] = score >= Eval;
				else
					comp[idx_comp] = score <= Eval;

				bool c = comp[idx_comp];
				int stop=1;
			}
			for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
			{
#pragma HLS unroll

				if(idx_comp==0)
					if(idx_topk==0)
						Eprev[idx_comp] = E[idx_comp];
					else
						Eprev[idx_comp] = E_prev_reg;
				else
					Eprev[idx_comp] = E[idx_comp-1];

				int eeevaalPrev_reg = (int)((ap_int<8>)E_prev_reg.range(DATABITS-1,0));
				int eeevaalPrev = (int)((ap_int<8>)E[idx_comp].range(DATABITS-1,0));

				int IDeeevaalPrev_reg = (int)((ap_int<8>)E_prev_reg.range(BUFFERBITS-1,DATABITS));
				int IDeeevaalPrev = (int)((ap_int<8>)E[idx_comp].range(BUFFERBITS-1,DATABITS));

				int stop=1;
			}

			bool comp_prev_val;
			if(idx_topk==0)
				comp_prev_val = 0;
			else
				comp_prev_val = comp_prev_reg;

			//fprintf(stderr,"\n");
			for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
			{
#pragma HLS unroll

				//				int c_prev = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(DATABITS-1,0));

				if(idx_comp!=0)
					TopKBuffer[idx_comp][idx_topk] = (comp[idx_comp] xor comp[idx_comp-1]) ? score_idx : ((comp[idx_comp]) ? Eprev[idx_comp] : E[idx_comp]);
				else
					TopKBuffer[idx_comp][idx_topk] = (comp[idx_comp] xor comp_prev_val) ? score_idx : ((comp[idx_comp]) ? Eprev[idx_comp] : E[idx_comp]);

				//								int a = idx_comp;
				//								int b = idx_topk;
				//								int c = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(DATABITS-1,0));
				//								int d = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(BUFFERBITS-1,DATABITS));
				//								fprintf(stderr,"\tTopKBuffer[%d][%d] = %d,%d",idx_comp,idx_topk,c,d);
			}

			E_prev_reg = E[NUM_COMP-1];
			comp_prev_reg = comp[NUM_COMP-1];

			int c3 = (int)E_prev_reg.range(DATABITS-1,0);
			int stop=0;

			//	TopKBuffer[idx_topk] = CompMux<TOPK,SORTORDER>(score, &E_prev_reg, &C_prev_reg, TopKBuffer[idx_topk], idx_topk);

		}//idx_topk
	}//idx_in
#endif
	//	fprintf(stderr,"\n****************************************************************");
	//	for(int idx_topk=0;idx_topk<(BUFFERDEPTH);idx_topk++){
	//		for(int idx_comp=0; idx_comp< NUM_COMP; idx_comp++)
	//		{
	//#pragma HLS unroll
	//
	//			int a = idx_comp;
	//			int b = idx_topk;
	//			int c = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(DATABITS-1,0));
	//			int d = (int)((ap_int<8>)TopKBuffer[idx_comp][idx_topk].range(BUFFERBITS-1,DATABITS));
	//			fprintf(stderr,"\nTopKBuffer[%d][%d] = %d,%d",idx_comp,idx_topk,c,d);
	//		}
	//	}
	//	int stop=0;

	//	LW:for(short i=0;i<TOPK;i++){
	//	#pragma HLS PIPELINE II=1
	//			short topk_id = i;
	//			ap_uint<BUFFERBITS> datatmp = TopKBuffer[topk_id%NUM_COMP][topk_id/NUM_COMP];
	//			dstStrm.write(datatmp);
	//	}

	short index=0;
	LW1:for(ap_uint<8> d2=0;d2<BUFFERDEPTH;d2++){
		LW2:for(ap_uint<8> d1=0;d1<NUM_COMP;d1++){
#pragma HLS PIPELINE II=1
			ap_uint<BUFFERBITS> datatmp = TopKBuffer[d1][d2];
			if(index<TOPK)
				dstStrm.write(datatmp);
			index++;
		}

	}
}
#endif

//# Disabled for NMS integration
#if 0
template <int DATABITS, int OUT_PTR_WIDTH>
void ReadWrite_box(ap_uint<OUT_PTR_WIDTH>* src,
		ap_uint<OUT_PTR_WIDTH>* dst,
		int inputSize_perclass
)
{
#pragma HLS inline off

   short words_per_cycle=OUT_PTR_WIDTH/DATABITS;
   short loop_bound = (((inputSize_perclass*4)+words_per_cycle-1)/words_per_cycle);
   L_BOX:for(int index=0 ; index < loop_bound; index++)
   {
#pragma HLS pipeline II=1
	dst[index] = src[index];
   }

}
#endif

template <
int NUM_CLASS,
int NCPC,
int IN_PTR_WIDTH,
int TOPK,
int SORTORDER,
int NUM_COMP,
int BUFFERDEPTH,
int DATABITS,
int INDEXBITS,
int INDEXBITS_OUT,
int BUFFERBITS,
int OUT_PTR_WIDTH,
int OUTSTREAMBITS,
int OUTSTREAMBITS_SCORE,
int OUTSTREAMBITS_INDEX,
int DEPTH_SIZE_BUFF>
void Read_Process(
		ap_uint<IN_PTR_WIDTH>* src_conf,
		//ap_uint<OUT_PTR_WIDTH>* src_box,
		//ap_uint<OUT_PTR_WIDTH>* dst,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		ap_uint<INDEXBITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF],
		int inputSize_perclass)
{
//#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1
#pragma HLS inline off
	hls::stream<ap_int<DATABITS> > in_multiclass[NUM_CLASS];
#pragma HLS ARRAY_PARTITION variable=in_multiclass complete dim=1

	hls::stream<ap_uint<INDEXBITS> > class_size;

	ap_uint<24> loop_bound_process = inputSize_perclass*BUFFERDEPTH;
	//int loop_bound_read = inputSize_perclass*(NUM_CLASS_CEIL/NCPC);
#pragma  HLS dataflow

	Read_input<NUM_CLASS,NCPC, DATABITS, IN_PTR_WIDTH, SORTORDER, INDEXBITS, TOPK>(src_conf, in_multiclass,inputSize_perclass, class_size);

	for(int class_id=0; class_id<NUM_CLASS; class_id++)
	{
#pragma HLS unroll
		Sort_1class<TOPK, SORTORDER, NUM_COMP, BUFFERDEPTH, DATABITS, INDEXBITS, BUFFERBITS, OUT_PTR_WIDTH,OUTSTREAMBITS,OUTSTREAMBITS_SCORE,OUTSTREAMBITS_INDEX>( in_multiclass[class_id], inputSize_perclass,loop_bound_process, TopKBuffer[class_id]);
	}

	Write_outputBuffer_size<NUM_CLASS, TOPK, NUM_COMP, OUT_PTR_WIDTH, BUFFERDEPTH, DATABITS, INDEXBITS, INDEXBITS_OUT, BUFFERBITS, DEPTH_SIZE_BUFF>(class_size, outSize_of_class);

	//ReadWrite_box<DATABITS, OUT_PTR_WIDTH>(src_box, dst, inputSize_perclass);
}

template <int NUM_CLASS,
int NCPC,
int TOPK,
int SORTORDER,
int DATABITS,
int INDEXBITS,
int INDEXBITS_OUT,
int IN_PTR_WIDTH,
int OUT_PTR_WIDTH,
int OUT_PTR_WIDTH_INDEX,
int OUT_PTR_WIDTH_SIZE,
int NUM_CLASS_CEIL,
int NUM_COMP,
int BUFFERBITS,
int BUFFERDEPTH,
int DEPTH_SIZE_BUFF>
void Sort_Multiclass(ap_uint<IN_PTR_WIDTH>*  src_conf,
		//ap_uint<OUT_PTR_WIDTH>* src_box,
		//ap_uint<OUT_PTR_WIDTH>* dst,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		ap_uint<INDEXBITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF],
		int inputSize_perclass)
{
	enum{
		/*		NUM_CLASS_CEIL = ((NUM_CLASS + NCPC - 1)/NCPC)*NCPC,
		NUM_AVIL_CLKS  = int(NUM_CLASS_CEIL/NCPC), // Floor vaue
		NUM_COMP_TMP = int((TOPK + NUM_AVIL_CLKS-1)/NUM_AVIL_CLKS), // Ceil value
		NUM_COMP = (NUM_COMP_TMP>16 && NUM_COMP_TMP<=32)*32 + (NUM_COMP_TMP>8 && NUM_COMP_TMP<=16)*16 
				+ (NUM_COMP_TMP>4 && NUM_COMP_TMP<=8)*8 
				+ (NUM_COMP_TMP>2 && NUM_COMP_TMP<=4)*4 + (NUM_COMP_TMP==2 || NUM_COMP_TMP==1)*NUM_COMP_TMP,  
		 */
		 //BUFFERDEPTH = (TOPK + NUM_COMP - 1)/NUM_COMP,
		 //BUFFERBITS = DATABITS+INDEXBITS,
		 OUTSTREAMBITS = BUFFERBITS*NUM_COMP,
		 OUTSTREAMBITS_SCORE = DATABITS*NUM_COMP,
		 OUTSTREAMBITS_INDEX = INDEXBITS*NUM_COMP,
		 NUM_CLASS_CEIL_COMP = ((NUM_CLASS+NUM_COMP-1)/NUM_COMP)*NUM_COMP,
		 //DEPTH_SIZE_BUFF = NUM_CLASS_CEIL_COMP/NUM_COMP
	};

	assert((NUM_COMP<=32) &&
			"NUM_COMP > 32 is not supported, Reduce NCPC count");

	//	fprintf(stderr,"\n********* 8bit change ********\n");
	//	fprintf(stderr,"\n********* NNUM_COMP=%d ********\n",NUM_COMP);
	////	std::cout << "NUM_CLASS " << NUM_CLASS << std::endl;
	//	std::cout << "NCPC " << NCPC << std::endl;
	//	std::cout << "TOPK " << TOPK << std::endl;
	//	std::cout << "PTRWIDTH " << PTR_WIDTH << std::endl;
	//	std::cout << "COMP_NUM " << NUM_COMP << std::endl;
	//	std::cout << "n " << n << std::endl;
/*	ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH];
	ap_uint<INDEXBITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF];
#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1
#pragma HLS resource variable=TopKBuffer core=RAM_2P_LUTRAM
#pragma HLS resource variable=outSize_of_class core=RAM_S2P_LUTRAM
*/
	short loop_bound_wr_ConfIndexSize = NUM_CLASS*((BUFFERDEPTH/2)+BUFFERDEPTH)+DEPTH_SIZE_BUFF;

	//Read_Process<NUM_CLASS, NCPC, IN_PTR_WIDTH ,TOPK, SORTORDER, NUM_COMP, BUFFERDEPTH, DATABITS, INDEXBITS, INDEXBITS_OUT, BUFFERBITS, OUT_PTR_WIDTH,OUTSTREAMBITS,OUTSTREAMBITS_SCORE,OUTSTREAMBITS_INDEX, DEPTH_SIZE_BUFF>
	//(src_conf, src_box, dst+loop_bound_wr_ConfIndexSize, TopKBuffer, outSize_of_class, inputSize_perclass);

	Read_Process<NUM_CLASS, NCPC, IN_PTR_WIDTH ,TOPK, SORTORDER, NUM_COMP, BUFFERDEPTH, DATABITS, INDEXBITS, INDEXBITS_OUT, BUFFERBITS, OUT_PTR_WIDTH,OUTSTREAMBITS,OUTSTREAMBITS_SCORE,OUTSTREAMBITS_INDEX, DEPTH_SIZE_BUFF>
		(src_conf, TopKBuffer, outSize_of_class, inputSize_perclass);

	//Write_output<NUM_CLASS, TOPK, NUM_COMP, OUT_PTR_WIDTH, OUT_PTR_WIDTH_INDEX, BUFFERDEPTH, DATABITS, INDEXBITS, INDEXBITS_OUT, BUFFERBITS, OUTSTREAMBITS,OUTSTREAMBITS_SCORE,OUTSTREAMBITS_INDEX,DEPTH_SIZE_BUFF>(TopKBuffer, outSize_of_class, dst, loop_bound_wr_ConfIndexSize);//, class_size, dst_size);

}
} // namespace cv
} // namespace xf
#endif //_XF_SORT_HPP_
