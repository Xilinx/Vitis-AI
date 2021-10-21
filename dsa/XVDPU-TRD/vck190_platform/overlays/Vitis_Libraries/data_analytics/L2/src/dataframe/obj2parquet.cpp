
#include <hls_stream.h>
//#include "xf_data_analytics/dataframe/read_from_dataframe.hpp"
#include "xf_data_analytics/dataframe/write_to_parquet.hpp"
//#include "xf_data_analytics/etl/dataframe_to_parquet.hpp"

//#define _DF_DEBUG 1

extern "C" void ObjToParquet(ap_uint<88> ddr_obj[1 << 25], ap_uint<8> schema[16], ap_uint<64> ddr_parquet[1 << 25]) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 32 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_0 port = ddr_obj

#pragma HLS INTERFACE m_axi offset = slave latency = 32 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_1 port = schema

#pragma HLS INTERFACE m_axi offset = slave latency = 32 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_2 port = ddr_parquet


#pragma HLS INTERFACE s_axilite port = ddr_obj bundle = control
#pragma HLS INTERFACE s_axilite port = schema bundle = control
#pragma HLS INTERFACE s_axilite port = ddr_parquet bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    xf::data_analytics::dataframe::writeToParquetSupport(ddr_obj, schema, ddr_parquet);
}
