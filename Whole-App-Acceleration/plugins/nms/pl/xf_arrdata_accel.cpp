
#include "xf_arrdata_config.h"
#include "xf_arrdata.hpp"

void arrnmsdata_accel(
		ap_uint<PRIOR_PTR_WIDTH>* priors,
		ap_uint<IN_PTR_WIDTH>* inBoxes,
		ap_uint<OUT_PTR_WIDTH>* outBoxes,
		short int *sort_index,
		short int sort_size)
{
#pragma HLS INTERFACE m_axi      port=priors       	offset=slave  bundle=gmem0
#pragma HLS INTERFACE m_axi      port=inBoxes       	offset=slave  bundle=gmem1
#pragma HLS INTERFACE m_axi      port=outBoxes       	offset=slave  bundle=gmem2
#pragma HLS INTERFACE m_axi      port=sort_index       	offset=slave  bundle=gmem2
#pragma HLS INTERFACE s_axilite  port=sort_size
#pragma HLS INTERFACE s_axilite  port=return

	arradataTop(priors, inBoxes, outBoxes, sort_index, sort_size);

}
//}


