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


#include "xf_common_config.h"


template<int INBOX_WIDTH, int BUFFERBITS, int BUFFERDEPTH>
void data_arr_nms(
		ap_uint<GMEM_GL_PRIOR_PTR_WIDTH>* priors,
		ap_uint<INBOX_WIDTH>* inBoxes1,
		ap_uint<INBOX_WIDTH>* inBoxes2,
		ap_uint<INBOX_WIDTH>* inBoxes3,
		ap_uint<INBOX_WIDTH>* inBoxes4,
		ap_uint<GMEM_GL_OUT_PTR_WIDTH>* outBoxes,
		ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH],
		int cl_id,
		short int sort_size,
		short int nms_th) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_int<GMEM_GL_OUT_PTR_WIDTH> > nmsStrm_in;

#if GL_TEST
	arradataTop<GMEM_GL_PRIOR_PTR_WIDTH, OUT_PTR_WIDTH, GMEM_GL_OUT_PTR_WIDTH, GMEM_LUT_PTR_WIDTH, CONFIG_MAX_BOX_PER_CLASS, BUFFERBITS, BUFFERDEPTH>(
			priors,	inBoxes, outBoxes+out_off, TopKBuffer, sort_size, cl_id);
#else
	arradataTop<GMEM_GL_PRIOR_PTR_WIDTH, INBOX_WIDTH, GMEM_GL_OUT_PTR_WIDTH, GMEM_LUT_PTR_WIDTH, CONFIG_MAX_BOX_PER_CLASS, BUFFERBITS, BUFFERDEPTH>(
			priors,	inBoxes1, inBoxes2, inBoxes3, inBoxes4, nmsStrm_in, TopKBuffer, sort_size, cl_id);

	nmsTop<CONFIG_MAX_BOX_PER_CLASS, GMEM_GL_OUT_PTR_WIDTH, GMEM_GL_OUT_PTR_WIDTH>(
			nmsStrm_in,	outBoxes, sort_size, nms_th);
#endif
	return;

}

extern "C" {

void sort_nms_accel(
		ap_uint<IN_PTR_WIDTH>* inConf,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes1,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes2,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes3,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes4,
		ap_uint<GMEM_GL_PRIOR_PTR_WIDTH>* priors,
		ap_uint<GMEM_GL_OUT_PTR_WIDTH>* outBoxes,
		int inputSize_perclass,
		short int nms_th)
{
#pragma HLS INTERFACE m_axi      port=inConf       	offset=slave  bundle=gmem0
#pragma HLS INTERFACE m_axi      port=inBoxes1       	offset=slave  bundle=gmem1
#pragma HLS INTERFACE m_axi      port=inBoxes2      	offset=slave  bundle=gmem2
#pragma HLS INTERFACE m_axi      port=inBoxes3       	offset=slave  bundle=gmem3
#pragma HLS INTERFACE m_axi      port=inBoxes4       	offset=slave  bundle=gmem4
#pragma HLS INTERFACE m_axi      port=priors       	offset=slave  bundle=gmem5
#pragma HLS INTERFACE m_axi      port=outBoxes       	offset=slave  bundle=gmem6
#pragma HLS INTERFACE s_axilite  port=inputSize_perclass  
#pragma HLS INTERFACE s_axilite  port=nms_th 
#pragma HLS INTERFACE s_axilite  port=return 

	enum{
		BUFFERDEPTH = (TOPK + NUM_COMP - 1)/NUM_COMP,
		BUFFERBITS = DATA_BITS+INDEX_BITS,
		NUM_CLASS_CEIL_COMP = ((NUM_CLASS+NUM_COMP-1)/NUM_COMP)*NUM_COMP,
		DEPTH_SIZE_BUFF = NUM_CLASS_CEIL_COMP/NUM_COMP
	};

	ap_uint<BUFFERBITS*NUM_COMP> TopKBuffer[NUM_CLASS][BUFFERDEPTH];
	ap_uint<INDEX_BITS_OUT*NUM_COMP> outSize_of_class[DEPTH_SIZE_BUFF];
#pragma HLS ARRAY_PARTITION variable=TopKBuffer complete dim=1
#pragma HLS resource variable=TopKBuffer core=RAM_2P_LUTRAM
#pragma HLS resource variable=outSize_of_class core=RAM_S2P_LUTRAM

	xf::cv::Sort_Multiclass<NUM_CLASS, NCPC, TOPK, SORT_ORDER, DATA_BITS, INDEX_BITS,
	INDEX_BITS_OUT, IN_PTR_WIDTH,  OUT_PTR_WIDTH,  OUT_PTR_WIDTH_INDEX,
	OUT_PTR_WIDTH_SIZE, NUM_CLASS_CEIL, NUM_COMP,
	BUFFERBITS,	BUFFERDEPTH, DEPTH_SIZE_BUFF>
	( inConf, TopKBuffer, outSize_of_class, inputSize_perclass);

	short sort_size = 0;
	int out_off = 0;

	//std::cout << "index bits, num_comp, bufferbits: " << INDEX_BITS_OUT << ", " << NUM_COMP << ", " << BUFFERBITS <<"\n";
	for(int cl_id = 1; cl_id < NUM_CLASS; cl_id++)
	{
		int div_idx	= cl_id/NCPC;
		int pos_rem	= cl_id - NCPC*div_idx;

		ap_uint<INDEX_BITS_OUT*NUM_COMP> var = outSize_of_class[div_idx];
		sort_size = (short)var((pos_rem*16)+15, pos_rem*16);

		data_arr_nms<GMEM_BOX_IN_PTR_WIDTH, BUFFERBITS, BUFFERDEPTH>(priors, inBoxes1, inBoxes2, inBoxes3, inBoxes4,
				outBoxes+out_off, TopKBuffer, cl_id, sort_size, nms_th);


		out_off += CONFIG_MAX_BOX_PER_CLASS;
	}
	return;
} // End of kernel

} // End of extern C
