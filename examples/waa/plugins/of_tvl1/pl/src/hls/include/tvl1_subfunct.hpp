#include "hls_stream.h"
#include "common/xf_extra_utility.hpp"
#include "common/xf_common.hpp"
#include "core/xf_math.h"
#include <iostream>
#include "remapcubic_wts_Q2p14.hpp"

#define _SUBFUNCT_TPLT_DEC template<int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32>
#define _ESTDUAL_TPLT_DEC template<int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int PIN_DEPTH=2, int FLOW_BW>
#define _UPDATEU_TPLT_DEC template<int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int U_DEPTH=2, int FLOW_BW>
#define _ESTV_TPLT_DEC template<int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int U_DEPTH=2, int FLOW_BW>
#define _CALCGRAD_TPLT_DEC template<int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int I1W_DEPTH=2, int U_DEPTH=2>
#define _RNDSATURN_TPLT_DEC template <int IN_BW, int OUT_BW, int IN_FBITS, int OUT_FBITS, bool EN_RND = 1>
#define _DUPMAT2_TPLT_DEC  template<int TYPE = XF_16SC1, int ROWS = 224, int COLS = 224, int NPC = 1, int DEPTH1=2, int DEPTH2=2>
#define _DUPMAT4_TPLT_DEC  template<int TYPE = XF_16SC1, int ROWS = 224, int COLS = 224, int NPC = 1, int DEPTH1=2, int DEPTH2=2, int DEPTH3=2, int DEPTH4=2>
#define _MULFUNCT_TPLT_DEC template<int FLOW_T = XF_16SC1, int FLOW_FBITS = 8, int ROWS = 126, int COLS = 224, int NPC = 1, int SCALE_BW = 16>

namespace xf {
namespace cv {


// ======================================================================================
// A generic structure for REMAP operation
// --------------------------------------------------------------------------------------
// Template Args:-
//	SRC_T : Data type of source image element
//	DST_T : Data type of destination image element
//	FLOW_T : Data type of Flow vector element
//	MVF : Max Flow Value
//      ROWS : Image height
//      COLS : Image width
//      K_ROWS : filter height
//      K_COLS : filter width
//      NPPC : No.of pixels per clock
//      BORDER_T : Type of border to be used for edge pixel(s) computation
//		USE_URAM : use URAM resources for line buffer
// 		USE_MAT : use MAT
// ......................................................................................

// Some macros related to template (for easiness of coding)
#define _GENERIC_REMAP_TPLT_DEC template<typename F, int SRC_T, int DST_T, int FLOW_T, int MFV, int U_FBITS, int I_FBITS, int ROWS, int COLS, int K_ROWS, int K_COLS, int NPPC=1, int BORDER_T=XF_BORDER_REFLECT_101, int USE_URAM=0, int USE_MAT=1, int U_FIFO_DEPTH=2, int OUT_FIFO_DEPTH=2>
#define _GENERIC_REMAP_TPLT template<typename F, int SRC_T, int DST_T, int FLOW_T, int MFV, int U_FBITS, int I_FBITS,  int ROWS, int COLS, int K_ROWS, int K_COLS, int NPPC, int BORDER_T, int USE_URAM, int USE_MAT, int U_FIFO_DEPTH, int OUT_FIFO_DEPTH>
#define _GENERIC_REMAP GenericREMAP<F, SRC_T, DST_T, FLOW_T, MFV, U_FBITS, I_FBITS, ROWS, COLS, K_ROWS, K_COLS, NPPC, BORDER_T, USE_URAM, USE_MAT, U_FIFO_DEPTH, OUT_FIFO_DEPTH>

// Some global constants
#define CH_IDX_T     uint8_t
#define K_ROW_IDX_T  uint8_t
#define K_COL_IDX_T  uint8_t
#define COL_IDX_T    uint16_t // Support upto 65,535
#define ROW_IDX_T    uint16_t // Support upto 65,535
#define SIZE_IDX_T   uint32_t
#define BICUBIC_FILTER	4	//Bicubic Filter size

// Some internal constants
#define _NPPC           (XF_NPIXPERCYCLE(NPPC))                             // Number of pixel per clock to be processed
#define _NPPC_SHIFT_VAL (XF_BITSHIFT(NPPC))                                 // Gives log base 2 on NPPC; Used for shifting purpose in case of division
#define _ECPR           ((((MFV+BICUBIC_FILTER-2)+(_NPPC-1))/_NPPC))                       // Extra clocks required for processing a row

#define _DST_PIX_WIDTH  (XF_PIXELDEPTH(XF_DEPTH(DST_T, NPPC)))              // destination pixel width

_GENERIC_REMAP_TPLT_DEC class GenericREMAP {

public:
	// Internal regsiters/buffers
	xf::cv::Window<K_ROWS, XF_NPIXPERCYCLE(NPPC)+(K_COLS-1), XF_DTUNAME(SRC_T, NPPC)>     src_blk; // Kernel sized image block with pixel parallelism
	xf::cv::LineBuffer<K_ROWS-1, (COLS >> _NPPC_SHIFT_VAL), XF_TNAME(SRC_T, NPPC),
			(USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM), (USE_URAM ? K_ROWS-1 : 1)> buff;    // Line Buffer for K_ROWS from the image

	// Internal Registers
	COL_IDX_T        num_clks_per_row; // No.of clocks required for processing one row
	SIZE_IDX_T       rd_ptr_remap;           // Read pointer
	SIZE_IDX_T       rd_flow_ptr_remap;           // Read pointer
	SIZE_IDX_T       wr_ptr_remap;           // Write pointer

	// Default Constructor
	GenericREMAP() {num_clks_per_row=0; rd_ptr_remap=0; rd_flow_ptr_remap=0;wr_ptr_remap=0;}

	// Internal functions
	void initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src);

	void process_row_remap(ROW_IDX_T rin, ROW_IDX_T rout, xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src, xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Ux, xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Uy, xf::cv::Mat<DST_T, ROWS, COLS, NPPC, OUT_FIFO_DEPTH> &_dst);
	void process_image_remap(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src, xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Ux, xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Uy,xf::cv::Mat<DST_T, ROWS, COLS, NPPC, OUT_FIFO_DEPTH> &_dst);
};


// -----------------------------------------------------------------------------------
// Function to initialize internal registers and buffers
// -----------------------------------------------------------------------------------
_GENERIC_REMAP_TPLT void _GENERIC_REMAP::initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src) {
		// clang-format off
#pragma HLS inline
		// clang-format on

	// Computing no.of clocks required for processing a row of given image dimensions
	num_clks_per_row = (_src.cols + _NPPC - 1) >> _NPPC_SHIFT_VAL;

	// Read/Write pointer set to start location of input image
	rd_ptr_remap = 0;
	rd_flow_ptr_remap = 0;
	wr_ptr_remap = 0;

	return;
} // End of initialize()

_RNDSATURN_TPLT_DEC ap_int<OUT_BW> rounding_n_saturation(ap_int<IN_BW> input)
{
	ap_int<IN_BW> round_out;

	if(EN_RND)
	{
		ap_int<IN_BW> abs_in;
		if(input[IN_BW-1]==0)
			abs_in = input;
		else
			abs_in = -input;

		ap_int<IN_BW> add_tmp;
		ap_int<IN_BW> abs_out;

		if(IN_FBITS>=OUT_FBITS)
		{
			if(abs_in==(1<<(IN_FBITS - OUT_FBITS -1)))
				add_tmp = 0;
			else
				add_tmp = abs_in + (1<<(IN_FBITS - OUT_FBITS -1));

			abs_out = add_tmp>>(IN_FBITS - OUT_FBITS);      
		}
		else
		{
			add_tmp = abs_in;
			abs_out = add_tmp<<(OUT_FBITS-IN_FBITS);      
		}
		if(input[IN_BW-1]==0)
			round_out = abs_out;
		else
			round_out = -abs_out;
	}
	else
	{
		if(IN_FBITS>=OUT_FBITS)
		{
			round_out = input>>(IN_FBITS - OUT_FBITS);
		}
		else
		{
			round_out = input<<(OUT_FBITS - IN_FBITS);
		}

	}	

	ap_int<IN_BW> saturate_signed;
	if(round_out > ((1<<(OUT_BW-1)) -1))
		saturate_signed =  ((1<<(OUT_BW-1)) -1);
	else if(round_out < -(1<<(OUT_BW-1)))
		saturate_signed =  -(1<<(OUT_BW-1));
	else
		saturate_signed = round_out;

	return (ap_int<OUT_BW>)saturate_signed;

}


// -----------------------------------------------------------------------------------
// Function to process a row
// -----------------------------------------------------------------------------------
_GENERIC_REMAP_TPLT void _GENERIC_REMAP::process_row_remap(
		ROW_IDX_T rin,
		ROW_IDX_T rout,
		xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Ux,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Uy,
		xf::cv::Mat<DST_T, ROWS, COLS, NPPC, OUT_FIFO_DEPTH> &_dst) {
		// clang-format off
#pragma HLS inline
		// clang-format on

	// --------------------------------------
	// Constants
	// --------------------------------------
	const uint32_t _TC = (COLS >> _NPPC_SHIFT_VAL)+(K_COLS >> 1); // MAX Trip Count per row
	const int CHANNEL_NUM = XF_CHANNELS(FLOW_T, NPPC);
	const int FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPPC)/CHANNEL_NUM;

	// --------------------------------------
	// Internal variables
	// --------------------------------------
	// Loop count variable
	COL_IDX_T col_loop_cnt = num_clks_per_row + _ECPR;
	COL_IDX_T cout = 0;

	// To store out pixels in packed format
	XF_TNAME(DST_T, NPPC) out_pixels;
	XF_TNAME( SRC_T, NPPC) in_pixel;
	XF_TNAME( FLOW_T, NPPC) in_Ux;
	XF_TNAME( FLOW_T, NPPC) in_Uy;

	// --------------------------------------
	// Initialize source block buffer to all zeros
	// --------------------------------------
	SRC_INIT_LOOP:
	for (K_ROW_IDX_T kr=0; kr<K_ROWS; kr++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on

		for(K_COL_IDX_T kc=0; kc<(_NPPC + K_COLS - 1); kc++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
			src_blk.val[kr][kc] = 0;
		}
	}

	// --------------------------------------
	// Process columns of the row
	// --------------------------------------
	COL_LOOP:
	for (COL_IDX_T c = 0; c < col_loop_cnt; c++) {
		// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
#pragma HLS DEPENDENCE variable=buff.val inter false
		// clang-format on

		// Fetch next pixel of current row
		// .........................................................
		in_pixel = ((rin<_src.rows) && (c < num_clks_per_row)) ? _src.read(rd_ptr_remap++) : (XF_TNAME(SRC_T, NPPC))0;
		in_Ux = (c >= _ECPR) ? _Ux.read(rd_flow_ptr_remap) : (XF_TNAME(FLOW_T, NPPC))0;
		in_Uy = (c >= _ECPR) ? _Uy.read(rd_flow_ptr_remap++) : (XF_TNAME(FLOW_T, NPPC))0;
		//rd_ptr_remap++;

		// Fetch data from RAMs and store in 'src_blk' for processing
		// .........................................................
		BUFF_RD_LOOP:
		for (K_ROW_IDX_T kr=0; kr<K_ROWS; kr++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on

			XF_TNAME(SRC_T, NPPC) tmp_rd_buff;

			// Read packed data
			tmp_rd_buff = (kr==(K_ROWS-1)) ? in_pixel : (c < num_clks_per_row) ? buff.val[kr][c] : (XF_TNAME(SRC_T, NPPC))0;

			// Extract pixels from packed data and store in 'src_blk'
			xfExtractPixels<NPPC, XF_WORDWIDTH(SRC_T, NPPC), XF_DEPTH(SRC_T, NPPC)> ((XF_PTNAME(XF_DEPTH(SRC_T, NPPC)) *)src_blk.val[kr], tmp_rd_buff, (K_COLS-1));
		}


		// Process the kernel block
		// ........................
		PROCESS_BLK_LOOP:
		for (int pix_idx = 0, bit = 0; pix_idx < _NPPC; pix_idx++, bit+=FLOW_BW) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on

			XF_DTUNAME(DST_T, NPPC) out_pix;

			XF_DTUNAME(SRC_T, NPPC) NxM_src_blk[K_ROWS][K_COLS];
		// clang-format off
#pragma HLS ARRAY_PARTITION variable=NxM_src_blk complete dim=1
#pragma HLS ARRAY_PARTITION variable=NxM_src_blk complete dim=2
		// clang-format on

			// Extract _NPPC, NxM-blocks from 'src_blk'
			REARRANGE_LOOP:
			for (K_ROW_IDX_T krow=0; krow<K_ROWS; krow++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
				for (K_COL_IDX_T kcol=0; kcol<K_COLS; kcol++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
					NxM_src_blk[krow][kcol] = src_blk.val[krow][pix_idx + kcol];

				}
			}

			XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) ux_value = in_Ux.range(bit+FLOW_BW-1, bit);
			XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) uy_value = in_Uy.range(bit+FLOW_BW-1, bit);

			ROW_IDX_T Yout_idx = rout;
			COL_IDX_T Xout_idx = cout + pix_idx;

			// Apply the filter on the NxM_src_blk
			const int max_flow_value = MFV;
			F oper;
			oper.apply_remap(NxM_src_blk, ux_value, uy_value, Xout_idx, Yout_idx, out_pix, wr_ptr_remap, max_flow_value);

			// Start packing the out pixel value every clock of NPPC
			out_pixels.range(((pix_idx+1)*_DST_PIX_WIDTH)-1, (pix_idx*_DST_PIX_WIDTH)) = out_pix;
		}

		// Write the data out to DDR
		// .........................
		if (c >= _ECPR) {
			_dst.write(wr_ptr_remap++, out_pixels);
			cout+=_NPPC;
		}

		// Move the data in Line Buffers
		// ...........................................
		if (c < num_clks_per_row) {
			BUFF_WR_LOOP:
			for (K_ROW_IDX_T kr=0; kr<K_ROWS-1; kr++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
				buff.val[kr][c] = src_blk.val[kr+1][K_COLS-1];
			}
		}

		// Now get ready for next cycle of coputation. So copy the last K_COLS-1 data to start location of 'src_blk'
		// ...........................................
		SHIFT_LOOP:
		for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
			for (K_COL_IDX_T kc = 0; kc < K_COLS-1; kc++) {
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
				src_blk.val[kr][kc] = src_blk.val[kr][_NPPC + kc];
			}
		}
	}

	return;
} // End of process_row_remap()


// -----------------------------------------------------------------------------------
// Main function that runs the filter over the image
// -----------------------------------------------------------------------------------
_GENERIC_REMAP_TPLT void _GENERIC_REMAP::process_image_remap(
		xf::cv::Mat<SRC_T, ROWS, COLS, NPPC> &_src,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Ux,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPPC, U_FIFO_DEPTH> &_Uy,
		xf::cv::Mat<DST_T, ROWS, COLS, NPPC, OUT_FIFO_DEPTH> &_dst) {
		// clang-format off
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable=cubic_wts cyclic factor=16 dim=1 partition
		// clang-format on

	// Constant declaration
	const uint32_t _TC = ((COLS >> _NPPC_SHIFT_VAL)+(K_COLS >> 1))/NPPC; // MAX Trip Count per row considering N-Pixel parallelsim

	// ----------------------------------
	// Start process with initialization
	// ----------------------------------
	initialize(_src);

	// ----------------------------------
	// Initialize Line Buffer
	// ----------------------------------
	// Part1: Initialize the buffer with 1st (kernel height)/2 rows of image
	//        Start filling rows from (kernel height)/2 and rest depending on border type
	READ_LINES_INIT:
	for (K_ROW_IDX_T r=(MFV+1), rinn=0; r<(K_ROWS-1); r++, rinn++) { // Note: Ignoring last row
		for (COL_IDX_T c=0; c<num_clks_per_row; c++) {
		// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
		// clang-format on
			buff.val[r][c] = _src.read(rd_ptr_remap++); // Reading the rows of image
		}
	}
	// Part2: Take care of borders depending on border type.
	//        In border replicate mode, fill with 1st row of the image.
	BORDER_INIT:
	for (K_ROW_IDX_T r=0; r<(MFV+1); r++) {
		for (COL_IDX_T c=0; c<num_clks_per_row; c++) {
		// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
		// clang-format on
			buff.val[r][c] = (BORDER_T==XF_BORDER_REPLICATE) ? buff.val[(MFV+1)][c] : (XF_TNAME(SRC_T, NPPC)) 0;
		}
	}

	// ----------------------------------
	// Processing each row of the image
	// ----------------------------------
	ROW_LOOP:
	for (ROW_IDX_T rin = (MFV+BICUBIC_FILTER-2), rout = 0; rout < _dst.rows; rin++, rout++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
		// clang-format on

		process_row_remap(rin, rout, _src, _Ux, _Uy, _dst);
	}

	return;
} // End of process_image_remap()

// ======================================================================================

// ======================================================================================
// Class for REMAP computation
// ======================================================================================

template <int SRC_T, int DST_T, int FLOW_T, int U_FBITS, int I_FBITS, int K_ROWS, int K_COLS, int NPPC>
class REMAP {

public:


	// -------------------------------------------------------------------------
	// Creating apply function (applying remap bicubic interpolation)
	// -------------------------------------------------------------------------
	void apply_remap(XF_DTUNAME(SRC_T, NPPC) patch[K_ROWS][K_COLS], XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) Ux, XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) Uy, COL_IDX_T xIDX, ROW_IDX_T yIDX, XF_DTUNAME(DST_T, NPPC) &pix, int wr_ptr_remap, const int max_flow_value) {
		// clang-format off
#pragma HLS inline off
		// clang-format on


		const int NUM_CH = XF_CHANNELS(SRC_T, NPPC);
		const int IN_BW = XF_PIXELWIDTH(SRC_T, NPPC);

		XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) Max_Ux = max_flow_value<<U_FBITS;
		XF_PTNAME(XF_DEPTH(FLOW_T, NPPC)) Max_Uy = max_flow_value<<U_FBITS;

		if(Ux > Max_Ux){
			Ux = Max_Ux;
		}else if(Ux < -Max_Ux){
			Ux = -Max_Ux;
		}	

		if(Uy > Max_Uy){
			Uy = Max_Uy;
		}else if(Uy < -Max_Uy){
			Uy = -Max_Uy;
		}	

		ap_int<32> cx = (ap_int<32>)Ux + ( ((ap_int<32>)xIDX)<<U_FBITS) ;
		ap_int<32> cy = (ap_int<32>)Uy + ( ((ap_int<32>)yIDX)<<U_FBITS) ;

		ap_int<32> cx_Qxp5 = rounding_n_saturation<32,32,U_FBITS, 5>(cx);
		ap_int<32> cy_Qxp5 = rounding_n_saturation<32,32,U_FBITS, 5>(cy);

		ap_uint<5> cx_fbits = cx_Qxp5.range(4,0);
		ap_uint<5> cy_fbits = cy_Qxp5.range(4,0);
		short wts_offset = (cy_fbits*32) + cx_fbits;

		XF_PTNAME(XF_DEPTH(SRC_T, NPPC)) patch_extract_Q8p0[(max_flow_value*2) + 1][(max_flow_value*2) + 1][16];
		ap_int<16> patch_extract_Q9p7[(max_flow_value*2) + 1][(max_flow_value*2) + 1][16];
		// clang-format off
#pragma HLS ARRAY_PARTITION variable=patch_extract_Q8p0 dim=0
#pragma HLS ARRAY_PARTITION variable=patch_extract_Q9p7 dim=0
		// clang-format on

		for(int sy_idx=0; sy_idx<(max_flow_value*2) + 1;sy_idx++)
		{
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
			for(int sx_idx=0; sx_idx<(max_flow_value*2) + 1;sx_idx++)
			{
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
				for(ap_uint<8> acc_cnt=0;acc_cnt<16;acc_cnt++)
				{
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
					ap_uint<8> x_cnt = acc_cnt.range(1,0);
					ap_uint<8> y_cnt = acc_cnt.range(3,2);
					XF_PTNAME(XF_DEPTH(SRC_T, NPPC)) pixel_read = patch[y_cnt+sy_idx][x_cnt+sx_idx].range(0 + IN_BW - 1, 0);
					XF_PTNAME(XF_DEPTH(SRC_T, NPPC)) pixel_read_Qxp7 = pixel_read>>(I_FBITS-7);
					patch_extract_Q9p7[sy_idx][sx_idx][acc_cnt] = (ap_int<16>)(pixel_read_Qxp7);

				}
			}
		}

		ap_int<32> posx_window = (ap_int<32>)Max_Uy + (ap_int<32>)Ux ;
		ap_int<32> posy_window = (ap_int<32>)Max_Uy + (ap_int<32>)Uy ;

		unsigned int Yidx_Qxp5 = rounding_n_saturation<32,32,U_FBITS, 5>(posy_window);
		unsigned int Xidx_Qxp5 = rounding_n_saturation<32,32,U_FBITS, 5>(posx_window);
		unsigned int Yidx_t = Yidx_Qxp5>>5;
		unsigned int Xidx_t = Xidx_Qxp5>>5;

		ap_int<48> sum1=0;
		ap_int<48> sum2=0;
		for(ap_uint<8> acc_cnt=0, w_bits=0;acc_cnt<16;acc_cnt++, w_bits+=16)
		{
		// clang-format off
#pragma HLS UNROLL
		// clang-format on
			ap_uint<8> x_cnt = acc_cnt.range(1,0);
			ap_uint<8> y_cnt = acc_cnt.range(3,2);
			ap_int<16> weight_t = cubic_wts[(wts_offset*16) + acc_cnt];
			ap_int<16> input_Q9p7 = patch_extract_Q9p7[Yidx_t][Xidx_t][acc_cnt];
			sum2+=input_Q9p7*weight_t;

		}

		XF_PTNAME(XF_DEPTH(SRC_T, NPPC)) out_t;
		out_t = rounding_n_saturation<48,IN_BW,14, 0>(sum2<<(I_FBITS-7));
		pix.range(0 + IN_BW - 1, 0) = out_t;

		return;
	}

};

// ======================================================================================
// Top REMAP API
// --------------------------------------------------------------------------------------
// Template Args:-
//         IN_TYPE : Data type of source image element
//         OUT_TYPE : Data type of destination image element
//         FLOW_TYPE : Data type of flow element
//         ROWS : Image height
//         COLS : Image width
//         K_SIZE : Bicubic window size
//         NPPC : No.of pixels per clock
//     	   BORDER_T : Type of border to be used for edge pixel(s) computation
//                (XF_BORDER_REPLICATE, XF_BORDER_CONSTANT, XF_BORDER_REFLECT_101, XF_BORDER_REFLECT)
//         USE_URAM : URAM use enable       
//         MAXFLOWVALUE : Max Flow Value       
//         FLOW_FBITS : Fraction bits of flow elements       
//         IMG_FBITS : Fraction bits of flow elements       
//         U_FIFODEPTH : stream FIFO depth of flow elements       
//         OUT_FIFODEPTH : stream FIFO depth of output elements       
// ......................................................................................
#define _REMAP_ REMAP<IN_TYPE, OUT_TYPE, FLOW_TYPE, FLOW_FBITS, IMG_FBITS, K_SIZE, K_SIZE, NPPC>

// --------------------------------------------------------------------------------------
// Function to compute REMAP using Bicubic interpotaion
// --------------------------------------------------------------------------------------
template<int IN_TYPE,int OUT_TYPE, int FLOW_TYPE, int ROWS, int COLS, int K_SIZE, int NPPC=1, int BORDER_T=XF_BORDER_REFLECT_101, int USE_URAM=0, int MAXFLOWVALUE, int FLOW_FBITS, int IMG_FBITS, int U_FIFODEPTH=2, int OUT_FIFODEPTH=2>
void Remap_Bicubic (xf::cv::Mat<IN_TYPE, ROWS, COLS, NPPC> &_src,
		xf::cv::Mat<FLOW_TYPE, ROWS, COLS, NPPC, U_FIFODEPTH> &_Ux,
		xf::cv::Mat<FLOW_TYPE, ROWS, COLS, NPPC, U_FIFODEPTH> &_Uy,
		xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPPC, OUT_FIFODEPTH> &_dst) {
		// clang-format off
#pragma HLS inline off
		// clang-format on

	xf::cv::GenericREMAP<_REMAP_, IN_TYPE, OUT_TYPE, FLOW_TYPE, MAXFLOWVALUE, FLOW_FBITS, IMG_FBITS, ROWS, COLS, K_SIZE, K_SIZE, NPPC, BORDER_T, USE_URAM, 1, U_FIFODEPTH, OUT_FIFODEPTH> remap;

	remap.process_image_remap(_src, _Ux, _Uy, _dst);

	return;
}


#undef _REMAP_

_CALCGRAD_TPLT_DEC int calcgrad(
		xf::cv::Mat<IMG_T, ROWS, COLS, NPC>& I0,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, I1W_DEPTH>& I1w,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& I1wx,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& I1wy,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, U_DEPTH>& U1,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, U_DEPTH>& U2,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& grad,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& rhoc,
		unsigned short height,
		unsigned short width_ncpr) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW
	};

	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_grad, pxl_pack_out_rhoc;
	XF_SNAME(WORDWIDTH_SRC1) pxl_pack_in_I0;
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_I1wx, pxl_pack_in_I1wy, pxl_pack_in_I1w;
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_U1, pxl_pack_in_U2;

	int read_ptr = 0 ;
	int write_ptr = 0 ;
	RowLoop:
	for (short i = 0; i < height; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		// clang-format on
		ColLoop:
		for (short j = 0; j < width_ncpr; j++) {
			// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline II=1
			// clang-format on

			//Read input stream
			pxl_pack_in_I0   = (XF_SNAME(WORDWIDTH_SRC1))(  I0.read(read_ptr));
			pxl_pack_in_I1w  = (XF_SNAME(WORDWIDTH_SRC2))( I1w.read(read_ptr));
			pxl_pack_in_I1wx = (XF_SNAME(WORDWIDTH_SRC2))(I1wx.read(read_ptr));
			pxl_pack_in_I1wy = (XF_SNAME(WORDWIDTH_SRC2))(I1wy.read(read_ptr));
			pxl_pack_in_U1   = (XF_SNAME(WORDWIDTH_SRC2))(  U1.read(read_ptr));
			pxl_pack_in_U2   = (XF_SNAME(WORDWIDTH_SRC2))(  U2.read(read_ptr));
			read_ptr++;

			ProcLoop:
			for (int bit_in1 = 0, bit_in2 =0 , bit_out = 0; bit_in1 < ((IMG_BW << XF_BITSHIFT(NPC)) * PLANES); bit_in1 += IMG_BW,  bit_in2 += FLOW_BW ,bit_out += FLOW_BW) {
		// clang-format off
#pragma HLS unroll
		// clang-format on
				//Extract pixels
				XF_PTNAME(DEPTH_SRC1) I0_pxl 	= pxl_pack_in_I0.range(  bit_in1 + IMG_BW - 1, bit_in1);
				XF_PTNAME(DEPTH_SRC2) I1w_pxl  	= pxl_pack_in_I1w.range( bit_in2 + FLOW_BW - 1, bit_in2);
				XF_PTNAME(DEPTH_SRC2) I1wx_pxl 	= pxl_pack_in_I1wx.range(bit_in2 + FLOW_BW- 1, bit_in2);
				XF_PTNAME(DEPTH_SRC2) I1wy_pxl 	= pxl_pack_in_I1wy.range(bit_in2 + FLOW_BW- 1, bit_in2);
				XF_PTNAME(DEPTH_SRC2) U1_pxl 	= pxl_pack_in_U1.range(  bit_in2 + FLOW_BW- 1, bit_in2);
				XF_PTNAME(DEPTH_SRC2) U2_pxl 	= pxl_pack_in_U2.range(  bit_in2 + FLOW_BW- 1, bit_in2);

				//*********  Computation:
				//*********  grad(i,j) = (I_1wx (i,j))^2 + (I_1wy (i,j))^2
				ap_int<INT_BW> I1wx_pow_2 = I1wx_pxl * I1wx_pxl; 		
				ap_int<INT_BW> I1wy_pow_2 = I1wy_pxl * I1wy_pxl; 		
				ap_int<INT_BW> grad_t1 	  = I1wx_pow_2 + I1wy_pow_2;	

				ap_int<FLOW_BW> grad_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS>(grad_t1);
				pxl_pack_out_grad.range(bit_out + FLOW_BW - 1, bit_out) = grad_t2;
				
				//*********  Computation:
				//*********  rho_c (i,j) = I_1w (i,j) - I_1wx (i,j) * u1(i,j) - I_1wy (i,j) * u2(i,j) - I_0 (i,j)
				ap_int<INT_BW> mul1  = I1wx_pxl * U1_pxl; 
				ap_int<INT_BW> mul2  = I1wy_pxl * U2_pxl; 
				ap_int<INT_BW> I1w_t = ((ap_int<INT_BW>)I1w_pxl)<<(FLOW_FBITS); 		
				ap_int<INT_BW> I0_t  = ((ap_int<INT_BW>)I0_pxl)<<(2*FLOW_FBITS - IMG_FBITS);  	
				ap_int<INT_BW> rhoc_t1= I1w_t - mul1 - mul2 - I0_t;  	

				ap_int<FLOW_BW> rhoc_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS>(rhoc_t1);
				pxl_pack_out_rhoc.range(bit_out + FLOW_BW - 1, bit_out) = rhoc_t2;//out Q8.8
				
			}

			// writing into output stream
			grad.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_grad); 
			rhoc.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_rhoc); 
			write_ptr++;
		}
	}
	return 0;
}


template<int IN1_BW, int IN2_BW, int OUT_BW, int IN1_FBITS, int IN2_FBITS, int OUT_FBITS, int DNR_FBITS>
ap_int<OUT_BW> fxdpt_div(ap_int<IN1_BW> in1, ap_int<IN2_BW> in2)
{
		// clang-format off
#pragma HLS inline
		// clang-format on

	ap_int<17> devisor_Q6p11 = rounding_n_saturation<IN2_BW, 17, IN2_FBITS, DNR_FBITS>(in2);;
	ap_uint<16> devisor_abs_Q5p11;
	if(devisor_Q6p11[16]==0)
		devisor_abs_Q5p11 = (devisor_Q6p11);
	else
		devisor_abs_Q5p11 = (-devisor_Q6p11);
	char div_fbits;
	ap_uint<32> inv_abs = (ap_uint<32>)Inverse((unsigned short)devisor_abs_Q5p11,  16-DNR_FBITS, &div_fbits);
	ap_uint<5> opt_div_fbits = div_fbits;
	ap_int<33> inv_val;
	if(devisor_Q6p11[16]==0)
		inv_val = inv_abs;
	else
		inv_val = -inv_abs;

	ap_int<IN1_BW+33> mul_out = inv_val * in1;
	ap_int<IN1_BW+33> mac_out = mul_out + (1<<(opt_div_fbits + IN1_FBITS - OUT_FBITS -1));
	ap_int<IN1_BW+33> out_tmp = mac_out >> (opt_div_fbits + IN1_FBITS - OUT_FBITS);
	ap_int<OUT_BW> div_out = out_tmp;

	ap_int<IN1_BW+33> mul_out_tmp = inv_val * in1;

	return div_out;

}

_ESTV_TPLT_DEC int estimateV(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& Iwx,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& Iwy,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_pass, 
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_pass, 
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& grad,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& rhoc,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& V1,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& V2,    
		ap_int<FLOW_BW> _lt,
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
		INT_BW_temp_rho_by_grad = 2*FLOW_FBITS + 2*FLOW_BW - FLOW_FBITS + 2*FLOW_FBITS - FLOW_FBITS,
		LTC = ROWS*TC
	};

	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_Iwx,pxl_pack_in_Iwy,pxl_pack_in_U1,pxl_pack_in_U2,pxl_pack_in_grad,pxl_pack_in_rhoc;    
	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_V1,pxl_pack_out_V2;    

	int read_ptr = 0 ;
	int write_ptr = 0 ;

	int i, j, k;

	int loopbound = height*width;
	int count = 0;
	RowColLoop:
	for (i = 0; i < loopbound; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=LTC max=LTC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		pxl_pack_in_Iwx  = (XF_SNAME(WORDWIDTH_SRC2))( Iwx.read(read_ptr)); 
		pxl_pack_in_Iwy  = (XF_SNAME(WORDWIDTH_SRC2))( Iwy.read(read_ptr)); 
		pxl_pack_in_U1   = (XF_SNAME(WORDWIDTH_SRC2))(  U1.read(read_ptr)); 
		pxl_pack_in_U2   = (XF_SNAME(WORDWIDTH_SRC2))(  U2.read(read_ptr)); 
		pxl_pack_in_grad = (XF_SNAME(WORDWIDTH_SRC2))(grad.read(read_ptr)); 
		pxl_pack_in_rhoc = (XF_SNAME(WORDWIDTH_SRC2))(rhoc.read(read_ptr)); 
		read_ptr++;

		ProcLoop:
		for (int bit_in1 = 0, bit_out = 0; bit_in1 < ((FLOW_BW << XF_BITSHIFT(NPC)) * PLANES); bit_in1 += FLOW_BW, bit_out += FLOW_BW) {
			// clang-format off
#pragma HLS UNROLL
			// clang-format on

			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) Iwx_pxl  = pxl_pack_in_Iwx.range( bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) Iwy_pxl  = pxl_pack_in_Iwy.range( bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U1_pxl   = pxl_pack_in_U1.range(  bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U2_pxl   = pxl_pack_in_U2.range(  bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) grad_pxl = pxl_pack_in_grad.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) rhoc_pxl = pxl_pack_in_rhoc.range(bit_in1 + FLOW_BW - 1, bit_in1);

			//*********  Computation:
			//*********  rho = rhoc(i,j) + ( Iwx(i,j)*u1(i,j) + Iwy(i,j)*u2(i,j) )
			ap_int<INT_BW> Iwx_U1_mul;
			ap_int<INT_BW> Iwy_U2_mul;
		// clang-format off
#pragma HLS BIND_OP variable=Iwx_U1_mul op=mul latency=3
#pragma HLS BIND_OP variable=Iwy_U2_mul op=mul latency=3
		// clang-format on
			Iwx_U1_mul = Iwx_pxl * U1_pxl;
			Iwy_U2_mul = Iwy_pxl * U2_pxl;
			ap_int<INT_BW> rho_INT_BW =  (((ap_int<INT_BW>)rhoc_pxl) << (FLOW_FBITS)) + (Iwx_U1_mul + Iwy_U2_mul); 
			ap_int<FLOW_BW> rho = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS,0>(rho_INT_BW);  
			ap_int<INT_BW>  ltInt_grad_INT_BW; 
		// clang-format off
#pragma HLS BIND_OP variable=ltInt_grad_INT_BW op=mul latency=3
		// clang-format on
			ltInt_grad_INT_BW = _lt*grad_pxl; 
			ap_int<FLOW_BW> ltInt_grad = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS,0>(ltInt_grad_INT_BW);   

			ap_int<FLOW_BW> mul_tmp_FLOW_TYPE = fxdpt_div<FLOW_BW, FLOW_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS,FLOW_FBITS,2>(rho, grad_pxl);

			ap_int<FLOW_BW> temp_rho_by_grad;
			if (grad_pxl)
				temp_rho_by_grad = (ap_int<FLOW_BW>)(mul_tmp_FLOW_TYPE); 
			else
				temp_rho_by_grad = 0x7FFF;

			ap_int<FLOW_BW> tmp_multiplier;

			if (rho < -ltInt_grad) {
				tmp_multiplier = _lt;
			} else if (rho > ltInt_grad) {
				tmp_multiplier = -_lt;
			} else if (grad_pxl > 0) {
				tmp_multiplier = -temp_rho_by_grad;
			} else {
				tmp_multiplier = 0;
			}
			ap_int<INT_BW> d1 , d2;
		// clang-format off
#pragma HLS BIND_OP variable=d1 op=mul latency=3
#pragma HLS BIND_OP variable=d2 op=mul latency=3
		// clang-format on
			d1 = tmp_multiplier * Iwx_pxl;
			d2 = tmp_multiplier * Iwy_pxl;
			
			ap_int<INT_BW> U1_d1_t1 = (((ap_int<INT_BW>)U1_pxl) << (FLOW_FBITS)) + d1;
			ap_int<INT_BW> U2_d2_t1 = (((ap_int<INT_BW>)U2_pxl) << (FLOW_FBITS)) + d2;

			ap_int<FLOW_BW> U1_d1_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS,0>(U1_d1_t1);    
			ap_int<FLOW_BW> U2_d2_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS,0>(U2_d2_t1);    

			pxl_pack_out_V1.range(bit_out + FLOW_BW - 1, bit_out) = U1_d1_t2;
			pxl_pack_out_V2.range(bit_out + FLOW_BW - 1, bit_out) = U2_d2_t2;
			
			count++;


		}
		// writing to output stream
		V1.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_V1);    
		V2.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_V2);       
		U1_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_in_U1);   
		U2_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_in_U2);   
		write_ptr++;

	}

	return 0;
}//estimateV

_SUBFUNCT_TPLT_DEC int divergence(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P1,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P2,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& divP,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P1_pass, 
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P2_pass, 
		unsigned short height,
		unsigned short width,bool intilize_P_with_zeros) {

		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
		NPPC = NPC
	};

	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_P1,pxl_pack_in_P2;   
	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_divP;   

	int read_ptr = 0 ;
	int write_ptr = 0 ;

	int i, j, k;
	ap_int<2*INT_BW> tmp_error = 0;
	// Line Buffer for 1 row from the image
	xf::cv::LineBuffer<1, (COLS >> _NPPC_SHIFT_VAL), XF_SNAME(WORDWIDTH_SRC2), RAM_S2P_BRAM, 1> line_buff;  

	XF_PTNAME(DEPTH_SRC2)    prev_P1_pxl = 0;
	XF_SNAME(WORDWIDTH_SRC2) prev_pxl_pack_in_P2 = 0;
	int row_count =0;
	int col_count =0;

	RowColLoop:
	for (int idx = 0; idx < height*width; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*TC max=ROWS*TC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		pxl_pack_in_P1 = (XF_SNAME(WORDWIDTH_SRC2))(       P1.read(read_ptr));    //Q8.8
		pxl_pack_in_P2 = (XF_SNAME(WORDWIDTH_SRC2))(       P2.read(read_ptr));    //Q8.8
		read_ptr++;
		prev_pxl_pack_in_P2 = (row_count==0)? (XF_SNAME(WORDWIDTH_SRC2))0 : line_buff(0,col_count);

		ProcLoop:
		for (int bit_in1 = 0, bit_out = 0; bit_in1 < ((FLOW_BW << XF_BITSHIFT(NPC)) * PLANES); bit_in1 += FLOW_BW, bit_out += FLOW_BW) {
			// clang-format off
#pragma HLS UNROLL
			// clang-format on

			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) P1_pxl_i_j         = pxl_pack_in_P1.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) P2_pxl_i_j         = pxl_pack_in_P2.range(bit_in1 + FLOW_BW - 1, bit_in1);

			XF_PTNAME(DEPTH_SRC2) P1_pxl_im1_j = (bit_in1 == 0) ? prev_P1_pxl : pxl_pack_in_P1.range(bit_in1 - 1, bit_in1 - FLOW_BW);
			XF_PTNAME(DEPTH_SRC2) P2_pxl_i_jm1 = prev_pxl_pack_in_P2.range(bit_in1 + FLOW_BW - 1, bit_in1);

			ap_int<INT_BW> v1x_pxl = P1_pxl_i_j - P1_pxl_im1_j;
			ap_int<INT_BW> v1y_pxl = P2_pxl_i_j - P2_pxl_i_jm1;
			ap_int<INT_BW> divP_pxl_t1 = v1x_pxl + v1y_pxl;

			ap_int<FLOW_BW> divP_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS>(divP_pxl_t1);    
			pxl_pack_out_divP.range(bit_out + FLOW_BW - 1, bit_out) = divP_pxl_t2;
			
		}
		if(col_count !=(width-1))
			prev_P1_pxl = pxl_pack_in_P1.range(FLOW_BW*NPC - 1, FLOW_BW*(NPC-1));
		else
			prev_P1_pxl = 0;

		line_buff(0,col_count) = pxl_pack_in_P2;

		XF_SNAME(WORDWIDTH_SRC2) pxl_pack_out_divP_wr;
		if(intilize_P_with_zeros==0)
			pxl_pack_out_divP_wr = pxl_pack_out_divP;
		else
			pxl_pack_out_divP_wr = 0;

		// writing to output stream
		divP.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_divP_wr);    
		P1_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_P1);    
		P2_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_P2);    
		write_ptr++;

		if(col_count == width-1)
		{
			col_count = 0;
			row_count++;
		}
		else
		{
			col_count++;
		}

	}
	
	return 0;
}//divergence

_UPDATEU_TPLT_DEC int updateU(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& V1,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& V2,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& divP1,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& divP2,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_in,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_in,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_out, 
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_out, 		
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_out_ddr,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_out_ddr,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P11_in,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P12_in,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P21_in,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P22_in,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P11_pass,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P12_pass,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P21_pass,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P22_pass,
		ap_int<ERR_BW> *error,
		ap_int<FLOW_BW> _theta,
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on

	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
	};

	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_V1,pxl_pack_in_V2,pxl_pack_in_divP1,pxl_pack_in_divP2,pxl_pack_in_U1_in,pxl_pack_in_U2_in;  
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_out_U1,pxl_pack_out_U2;  
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_p11,pxl_pack_in_p12,pxl_pack_in_p21,pxl_pack_in_p22;

	int read_ptr = 0 ;
	int write_ptr = 0 ;

	int i, j, k;
	ap_int<ERR_BW> tmp_errorInt_t1 = 0;

	RowColLoop:
	for (int idx = 0; idx < height*width; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*TC max=ROWS*TC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		pxl_pack_in_V1 = (XF_SNAME(WORDWIDTH_SRC2))(          V1.read(idx));   
		pxl_pack_in_V2 = (XF_SNAME(WORDWIDTH_SRC2))(          V2.read(idx));   
		pxl_pack_in_divP1 = (XF_SNAME(WORDWIDTH_SRC2))(    divP1.read(idx));   
		pxl_pack_in_divP2 = (XF_SNAME(WORDWIDTH_SRC2))(    divP2.read(idx));   
		pxl_pack_in_U1_in   = (XF_SNAME(WORDWIDTH_SRC2))(  U1_in.read(idx));   
		pxl_pack_in_U2_in   = (XF_SNAME(WORDWIDTH_SRC2))(  U2_in.read(idx));   
		pxl_pack_in_p11   = (XF_SNAME(WORDWIDTH_SRC2))(   P11_in.read(idx));   
		pxl_pack_in_p12   = (XF_SNAME(WORDWIDTH_SRC2))(   P12_in.read(idx));   
		pxl_pack_in_p21   = (XF_SNAME(WORDWIDTH_SRC2))(   P21_in.read(idx));   
		pxl_pack_in_p22   = (XF_SNAME(WORDWIDTH_SRC2))(   P22_in.read(idx));   

		ProcLoop:
		for (int bit_in1 = 0, bit_out = 0; bit_in1 < ((FLOW_BW << XF_BITSHIFT(NPC)) * PLANES); bit_in1 += FLOW_BW, bit_out += FLOW_BW) {
			// clang-format off
#pragma HLS UNROLL
			// clang-format on

			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) V1_pxl         = pxl_pack_in_V1.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) V2_pxl         = pxl_pack_in_V2.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) divP1_pxl     = pxl_pack_in_divP1.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) divP2_pxl     = pxl_pack_in_divP2.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U1_pxl         = pxl_pack_in_U1_in.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U2_pxl         = pxl_pack_in_U2_in.range(bit_in1 + FLOW_BW - 1, bit_in1);

			XF_PTNAME(DEPTH_SRC2) U1k_pxl,U2k_pxl;
			U1k_pxl = U1_pxl;
			U2k_pxl = U2_pxl;

			ap_int<INT_BW> U1_pxl_t1 = (((ap_int<INT_BW>)V1_pxl) << (FLOW_FBITS)) + _theta*divP1_pxl;
			ap_int<INT_BW> U2_pxl_t1 = (((ap_int<INT_BW>)V2_pxl) << (FLOW_FBITS)) + _theta*divP2_pxl;

			ap_int<INT_BW> U1_V1_pxl_t1 = (((ap_int<INT_BW>)(V1_pxl-U1_pxl)) << (FLOW_FBITS)) + _theta*divP1_pxl;
			ap_int<INT_BW> U2_V2_pxl_t1 = (((ap_int<INT_BW>)(V2_pxl-U2_pxl)) << (FLOW_FBITS)) + _theta*divP2_pxl;

			ap_int<FLOW_BW> U1_V1_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS, 0>(U1_V1_pxl_t1);
			ap_int<FLOW_BW> U2_V2_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS, 0>(U2_V2_pxl_t1);

			ap_int<INT_BW> U1_V1_pxl_t2_sq = U1_V1_pxl_t2*U1_V1_pxl_t2;
			ap_int<INT_BW> U2_V2_pxl_t2_sq = U2_V2_pxl_t2*U2_V2_pxl_t2;

			ap_int<INT_BW> add_U1_V1_U2_V2 = U1_V1_pxl_t2_sq + U2_V2_pxl_t2_sq;
			ap_int<ERR_BW> add_U1_V1_U2_V2_t1= rounding_n_saturation<INT_BW, ERR_BW, 2*FLOW_FBITS, ERR_BW/2, 0>(add_U1_V1_U2_V2);
			tmp_errorInt_t1 += add_U1_V1_U2_V2_t1;
			
			ap_int<FLOW_BW> U1_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS, 0>(U1_pxl_t1);   
			ap_int<FLOW_BW> U2_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS, 0>(U2_pxl_t1);   

			pxl_pack_out_U1.range(bit_in1 + FLOW_BW - 1, bit_in1) = U1_pxl_t2;
			pxl_pack_out_U2.range(bit_in1 + FLOW_BW - 1, bit_in1) = U2_pxl_t2;
			
		}
		// writing to output stream
		U1_out.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U1);    
		U2_out.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U2);      
		U1_out_ddr.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U1);  
		U2_out_ddr.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U2);  
		P11_pass.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_p11);    
		P12_pass.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_p12);    
		P21_pass.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_p21);    
		P22_pass.write(idx, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_in_p22);    
	}
	*error = tmp_errorInt_t1;

	return 0;
}//updateU

_SUBFUNCT_TPLT_DEC int forwardgradient(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1x,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1y,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P1,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P2,    
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P1_pass,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P2_pass,   
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
		NPPC = NPC
	};

	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_U1;    
	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_U1x,pxl_pack_out_U1y;   
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_P1,  pxl_pack_in_P2;    
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_out_P1,  pxl_pack_out_P2;  

	int read_ptr = 0 ;
	int write_ptr = 0 ;
	
	int i, j, k;
	ap_int<2*INT_BW> tmp_error = 0;
	xf::cv::LineBuffer<1, (COLS >> _NPPC_SHIFT_VAL), XF_SNAME(WORDWIDTH_SRC2), RAM_S2P_BRAM, 1> line_buff;      
	xf::cv::LineBuffer<1, (COLS >> _NPPC_SHIFT_VAL), XF_SNAME(WORDWIDTH_SRC2), RAM_S2P_BRAM, 1> line_buff_P1;   
	xf::cv::LineBuffer<1, (COLS >> _NPPC_SHIFT_VAL), XF_SNAME(WORDWIDTH_SRC2), RAM_S2P_BRAM, 1> line_buff_P2;   

	XF_SNAME(WORDWIDTH_SRC2) curr_pxl_pack_in_U1, next_pxl_pack_in_U1, pxl0_pack_in_U1;

	for (int c=0; c<width; c++) {
		// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=TC
		// clang-format on
		// Reading the rows of image
		line_buff(0, c) = U1.read(read_ptr); 
		line_buff_P1(0, c) = P1.read(read_ptr); 
		line_buff_P2(0, c) = P2.read(read_ptr++);
	}

	curr_pxl_pack_in_U1 = line_buff(0,0);
	pxl0_pack_in_U1 = 0;
	int col_count = 0;
	int row_count = 0;

	RowColLoop:
	for (int idx = 0; idx < height*width; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*TC max=ROWS*TC
#pragma HLS pipeline II=1
		// clang-format on
		//Read input stream
		pxl_pack_in_U1 = (row_count == height-1) ? curr_pxl_pack_in_U1 : (XF_SNAME(WORDWIDTH_SRC2))(U1.read(read_ptr));   
		if(row_count != (height-1))
		{
			pxl_pack_in_P1 = (XF_SNAME(WORDWIDTH_SRC2))(P1.read(read_ptr));   
			pxl_pack_in_P2 = (XF_SNAME(WORDWIDTH_SRC2))(P2.read(read_ptr));   
		}
		else
		{
			pxl_pack_in_P1 = 0;   
			pxl_pack_in_P2 = 0;   
		}
		read_ptr++;

		pxl_pack_out_P1 = line_buff_P1(0, col_count);
		pxl_pack_out_P2 = line_buff_P2(0, col_count);

		line_buff_P1(0, col_count) = pxl_pack_in_P1;
		line_buff_P2(0, col_count) = pxl_pack_in_P2;

		next_pxl_pack_in_U1 = (col_count==width-1) ? pxl0_pack_in_U1 : line_buff(0, col_count+1);

		line_buff(0, col_count) = pxl_pack_in_U1;

		if (col_count==0)
			pxl0_pack_in_U1 = pxl_pack_in_U1;

		ProcLoop:

		for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
			// clang-format off
#pragma HLS UNROLL
			// clang-format on

			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) U1_pxl_i_j = curr_pxl_pack_in_U1.range((k+1)*FLOW_BW - 1, k*FLOW_BW);
			XF_PTNAME(DEPTH_SRC2) tmp_U1_pxl_ip1_j = (k == (XF_NPIXPERCYCLE(NPC)-1)) ?
					next_pxl_pack_in_U1.range(FLOW_BW-1, 0) :
					curr_pxl_pack_in_U1.range((k+2)*FLOW_BW-1, (k+1)*FLOW_BW);
			XF_PTNAME(DEPTH_SRC2) U1_pxl_ip1_j = ((col_count == width-1) && (k == (XF_NPIXPERCYCLE(NPC)-1))) ? U1_pxl_i_j : tmp_U1_pxl_ip1_j;
			XF_PTNAME(DEPTH_SRC2) U1_pxl_i_jp1 = pxl_pack_in_U1.range((k+1)*FLOW_BW - 1, k*FLOW_BW);

			ap_int<INT_BW> U1x_pxl_t1 = U1_pxl_ip1_j - U1_pxl_i_j;
			ap_int<INT_BW> U1y_pxl_t1 = U1_pxl_i_jp1 - U1_pxl_i_j;
			ap_int<FLOW_BW> U1x_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS>(U1x_pxl_t1);   
			ap_int<FLOW_BW> U1y_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS>(U1y_pxl_t1);   
			pxl_pack_out_U1x.range((k+1)*FLOW_BW - 1, k*FLOW_BW) = U1x_pxl_t2;
			pxl_pack_out_U1y.range((k+1)*FLOW_BW - 1, k*FLOW_BW) = U1y_pxl_t2;

		}

		curr_pxl_pack_in_U1 = next_pxl_pack_in_U1;
		
		U1x.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U1x);      
		U1y.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_U1y);      
		P1_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_P1);   
		P2_pass.write(write_ptr, (XF_SNAME(WORDWIDTH_SRC2))pxl_pack_out_P2);   
		write_ptr++;

		if(col_count == width-1)
		{
			col_count = 0;
			row_count++;
		}
		else
		{
			col_count++;
		}

	}
	
	return 0;
}//forwardgradient

_SUBFUNCT_TPLT_DEC int centeredgradient(
		xf::cv::Mat<IMG_T, ROWS, COLS, NPC>& I1,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& I1x,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& I1y,	
		unsigned short height,
		unsigned short width_ncpr) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
		NPPC = NPC
	};

	unsigned short width = I1.cols;
	unsigned short valid_pix_last_col_NPC = (width_ncpr<<NPC) - width;
	XF_SNAME(WORDWIDTH_SRC1) pxl_pack_in_I1;   
	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_I1x,pxl_pack_out_I1y;    
	int read_ptr = 0 ;
	int write_ptr = 0 ;
	ap_uint<16> i, j, k;
	ap_int<2*INT_BW> tmp_error = 0;
	XF_SNAME(WORDWIDTH_SRC1) prev_line_pxl_pack_in_I1, curr_pxl_pack_in_I1, next_pxl_pack_in_I1, pxl0_pack_in_I1;
	XF_SNAME(WORDWIDTH_SRC1) prev_I1_pxl = 0;

	// Line Buffer for 1 row from the image
	xf::cv::LineBuffer<ROWS, (COLS >> _NPPC_SHIFT_VAL), XF_SNAME(WORDWIDTH_SRC1), RAM_S2P_BRAM, 1> line_buff;    
	// Intialize line buffer
	READ_LINES_INIT:
	for (int c=0; c< width_ncpr;c++) {
		// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=TC
		// clang-format on
		// Reading the rows of image
		line_buff(0,c) = I1.read(read_ptr++); 
	}
	curr_pxl_pack_in_I1 = line_buff(0,0);
	pxl0_pack_in_I1 = 0;

	RowLoop:
	for (i = 0; i < height; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		// clang-format on

		ColLoop:
		for (j = 0; j < (width_ncpr); j++) {
			// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline II=1
			// clang-format on

			//Read input stream
			pxl_pack_in_I1 = (i == height-1) ? curr_pxl_pack_in_I1 : (XF_SNAME(WORDWIDTH_SRC1))(I1.read(read_ptr++));    

			if (i[0] == 0) {
				prev_line_pxl_pack_in_I1 = line_buff(1, j);
				next_pxl_pack_in_I1 = (j==width_ncpr-1) ? pxl0_pack_in_I1 : line_buff(0, j+1);

				line_buff(1, j) = pxl_pack_in_I1;
			} else {
				prev_line_pxl_pack_in_I1 = line_buff(0,j);
				next_pxl_pack_in_I1 = (j==width_ncpr-1) ? pxl0_pack_in_I1 : line_buff(1, j+1) ;

				line_buff(0, j) = pxl_pack_in_I1;
			}

			if (j==0)
				pxl0_pack_in_I1 = pxl_pack_in_I1;

			ProcLoop:

			for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
				// clang-format off
#pragma HLS UNROLL
				// clang-format on

				//Extract right_pixel, left_pixel, bottom_pixel & top_pixel pixels for computation.
				unsigned short in_col_id = (j*XF_NPIXPERCYCLE(NPC) + k);
				bool en_last_pixel_in_row = ( in_col_id == (width-1));
				XF_PTNAME(DEPTH_SRC1) tmp_I1_pxl_ip1_j = (k == (XF_NPIXPERCYCLE(NPC)-1)) ?
						next_pxl_pack_in_I1.range(IMG_BW-1, 0) :
						curr_pxl_pack_in_I1.range((k+2)*IMG_BW-1, (k+1)*IMG_BW);
				XF_PTNAME(DEPTH_SRC1) tmp_I1_pxl_im1_j = (k==0) ? prev_I1_pxl : curr_pxl_pack_in_I1.range(k*IMG_BW-1, (k-1)*IMG_BW);
				XF_PTNAME(DEPTH_SRC1) I1_pxl_ip1_j = (en_last_pixel_in_row) ? curr_pxl_pack_in_I1.range((k+1)*IMG_BW - 1, k*IMG_BW) : tmp_I1_pxl_ip1_j;
				XF_PTNAME(DEPTH_SRC1) I1_pxl_im1_j = ((j==0) && (k==0)) ? curr_pxl_pack_in_I1.range(IMG_BW-1, 0)  : tmp_I1_pxl_im1_j;
				XF_PTNAME(DEPTH_SRC1) I1_pxl_i_jp1 = pxl_pack_in_I1.range((k+1)*IMG_BW - 1, k*IMG_BW);
				XF_PTNAME(DEPTH_SRC2) I1_pxl_i_jm1 = (i==0) ? curr_pxl_pack_in_I1.range((k+1)*IMG_BW - 1, k*IMG_BW) : prev_line_pxl_pack_in_I1.range((k+1)*IMG_BW - 1, k*IMG_BW);

				// computation.
				// dx = ( right_pixel - left_pixel ) / 2
				// dy = ( bottom_pixel - top_pixel ) / 2
				ap_int<INT_BW> I1x_pxl_t1 = ( (((ap_int<INT_BW>)I1_pxl_ip1_j)<<1) - (((ap_int<INT_BW>)I1_pxl_im1_j)<<1) )/2;
				ap_int<INT_BW> I1y_pxl_t1 = ( (((ap_int<INT_BW>)I1_pxl_i_jp1)<<1) - (((ap_int<INT_BW>)I1_pxl_i_jm1)<<1) )/2;

				// output round & saturation.
				ap_int<FLOW_BW> I1x_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, (IMG_FBITS+1), FLOW_FBITS>(I1x_pxl_t1); 
				ap_int<FLOW_BW> I1y_pxl_t2 = rounding_n_saturation<INT_BW, FLOW_BW, (IMG_FBITS+1), FLOW_FBITS>(I1y_pxl_t1); 

				// Padding for last invalid pixels of a row
				if(in_col_id > (width-1))
				{
					I1x_pxl_t2=0;
					I1y_pxl_t2=0;
				}

				// output data packing
				pxl_pack_out_I1x.range((k+1)*FLOW_BW - 1, k*FLOW_BW) = I1x_pxl_t2;
				pxl_pack_out_I1y.range((k+1)*FLOW_BW - 1, k*FLOW_BW) = I1y_pxl_t2;

			}
			prev_I1_pxl = curr_pxl_pack_in_I1.range(NPC*XF_PIXELWIDTH(IMG_T, NPC)-1, NPC*XF_PIXELWIDTH(IMG_T, NPC)-IMG_BW);
			curr_pxl_pack_in_I1 = next_pxl_pack_in_I1;

			//write output stream
			I1x.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_I1x);    
			I1y.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_I1y);    
			write_ptr++;
		}
	}

	return 0;
}//Centeredgradient

template<int NUMBITS_C=32>
ap_int<32> tvl_isqrt(ap_int<32> num){
		// clang-format off
#pragma HLS inline
		// clang-format on

	bool  compute;
	ap_int<32> res = 0;
	ap_int<32> bit = 1 << (NUMBITS_C-2); 
	// The second-to-top bit is set.
	// Same as ((unsigned) INT32_MAX + 1) / 2.

	compute = (bit <= num);

	for (uint8_t i=0; i < NUMBITS_C/2; i++) {
		// clang-format off
#pragma HLS unroll
		// clang-format on
		if (compute) {
			if (num >= res + bit) {
				num -= res + bit;
				res = (res >> 1) + bit;
			} else {
				res >>= 1;
			}
		}
		bit >>= 2;
		// "bit" starts at the highest power of four <= the argument.
		if (!compute)
			compute = (bit <= num);
	}
	return res;
}//tvl_isqrt

_ESTDUAL_TPLT_DEC int estimatedualvariables(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1x,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1y,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2x,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2y,  
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P11_in,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P12_in,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P21_in,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P22_in,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P11_out,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P12_out,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P21_out,   
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& P22_out,   
		ap_int<FLOW_BW> taut,
		bool intilize_P_with_zeros,
		unsigned short height,
		unsigned short width) {

		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(IMG_T, NPC),
		DEPTH_SRC1 = XF_DEPTH(IMG_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC1 = XF_WORDWIDTH(IMG_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IMG_BW = XF_PIXELWIDTH(IMG_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW,
	};

	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_U1x,pxl_pack_in_U1y,pxl_pack_in_U2x,pxl_pack_in_U2y;   
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_P11_in,pxl_pack_in_P12_in,pxl_pack_in_P21_in,pxl_pack_in_P22_in;   
	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_P11_out,pxl_pack_out_P12_out,pxl_pack_out_P21_out,pxl_pack_out_P22_out;   
	ap_int<INT_BW> pxl_pack_out_P11_out_t1,pxl_pack_out_P12_out_t1,pxl_pack_out_P21_out_t1,pxl_pack_out_P22_out_t1;   

	int rd_ptr = 0 ;
	int wr_ptr = 0 ;

	int i, j, k;
	int count = 0;

	//wr_ptr = 0 ;
	EstDualRowColLoop:
	for (int idx = 0; idx < height*width; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*TC max=ROWS*TC
#pragma HLS pipeline II=1
		// clang-format on

		pxl_pack_in_U1x = (XF_SNAME(WORDWIDTH_SRC2))(U1x.read(idx));  
		pxl_pack_in_U1y = (XF_SNAME(WORDWIDTH_SRC2))(U1y.read(idx));  
		pxl_pack_in_U2x = (XF_SNAME(WORDWIDTH_SRC2))(U2x.read(idx));  
		pxl_pack_in_U2y = (XF_SNAME(WORDWIDTH_SRC2))(U2y.read(idx));  
		pxl_pack_in_P11_in = (XF_SNAME(WORDWIDTH_SRC2))(P11_in.read(idx));  
		pxl_pack_in_P12_in = (XF_SNAME(WORDWIDTH_SRC2))(P12_in.read(idx));  
		pxl_pack_in_P21_in = (XF_SNAME(WORDWIDTH_SRC2))(P21_in.read(idx));  
		pxl_pack_in_P22_in = (XF_SNAME(WORDWIDTH_SRC2))(P22_in.read(idx));  

		ProcLoop:
		for (int npc_count = 0, bit_in1 = 0, bit_out = 0; npc_count < NPC; npc_count++, bit_in1 += FLOW_BW, bit_out += FLOW_BW) {
			// clang-format off
#pragma HLS UNROLL
			// clang-format on

			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) U1x_pxl    = pxl_pack_in_U1x.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U1y_pxl    = pxl_pack_in_U1y.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U2x_pxl    = pxl_pack_in_U2x.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) U2y_pxl    = pxl_pack_in_U2y.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) P11_in_pxl = pxl_pack_in_P11_in.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) P12_in_pxl = pxl_pack_in_P12_in.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) P21_in_pxl = pxl_pack_in_P21_in.range(bit_in1 + FLOW_BW - 1, bit_in1);
			XF_PTNAME(DEPTH_SRC2) P22_in_pxl = pxl_pack_in_P22_in.range(bit_in1 + FLOW_BW - 1, bit_in1);

			ap_int<INT_BW> U1x_pxl_sq = U1x_pxl*U1x_pxl;
			ap_int<INT_BW> U1y_pxl_sq = U1y_pxl*U1y_pxl;
			ap_int<INT_BW> U2x_pxl_sq = U2x_pxl*U2x_pxl;
			ap_int<INT_BW> U2y_pxl_sq = U2y_pxl*U2y_pxl;

			ap_int<INT_BW> sum_sq_U1 = U1x_pxl_sq+U1y_pxl_sq;
			ap_int<INT_BW> sum_sq_U2 = U2x_pxl_sq+U2y_pxl_sq;
			ap_int<32> sum_sq_1_Q16p16 = sum_sq_U1>>(2*FLOW_FBITS - 16);
			ap_int<32> sum_sq_2_Q16p16 = sum_sq_U2>>(2*FLOW_FBITS - 16);

			ap_int<32> g1_Q24p8 = tvl_isqrt(sum_sq_1_Q16p16);
			ap_int<32> g2_Q24p8 = tvl_isqrt(sum_sq_2_Q16p16);

			ap_int<18> g1_Q10p8 = (ap_int<18>)g1_Q24p8;
			ap_int<18> g2_Q10p8 = (ap_int<18>)g2_Q24p8;

			ap_int<INT_BW> one_INT_BW = 0; one_INT_BW[FLOW_FBITS+8]=1;
			ap_int<INT_BW> ng1_INT_BW = one_INT_BW + (taut*g1_Q10p8);
			ap_int<INT_BW> ng2_INT_BW = one_INT_BW + (taut*g2_Q10p8);

			ap_int<FLOW_BW> ng1 = ng1_INT_BW >> 8;
			ap_int<FLOW_BW> ng2 = ng2_INT_BW >> 8;

			ap_int<INT_BW> taut_U1x_t1 = taut * U1x_pxl;
			ap_int<INT_BW> taut_U1y_t1 = taut * U1y_pxl;
			ap_int<INT_BW> taut_U2x_t1 = taut * U2x_pxl;
			ap_int<INT_BW> taut_U2y_t1 = taut * U2y_pxl;

			ap_int<FLOW_BW> taut_U1x_t2 = taut_U1x_t1>>FLOW_FBITS;
			ap_int<FLOW_BW> taut_U1y_t2 = taut_U1y_t1>>FLOW_FBITS;
			ap_int<FLOW_BW> taut_U2x_t2 = taut_U2x_t1>>FLOW_FBITS;
			ap_int<FLOW_BW> taut_U2y_t2 = taut_U2y_t1>>FLOW_FBITS;

			ap_int<FLOW_BW> taut_U1x_t3,taut_U1y_t3;
			ap_int<FLOW_BW> taut_U2x_t3,taut_U2y_t3;
			if (intilize_P_with_zeros==true) {
				taut_U1x_t3 = taut_U1x_t2;
				taut_U1y_t3 = taut_U1y_t2;
				taut_U2x_t3 = taut_U2x_t2;
				taut_U2y_t3 = taut_U2y_t2;
			}
			else
			{
				taut_U1x_t3 = (P11_in_pxl + taut_U1x_t2) ;
				taut_U1y_t3 = (P12_in_pxl + taut_U1y_t2) ;
				taut_U2x_t3 = (P21_in_pxl + taut_U2x_t2) ;
				taut_U2y_t3 = (P22_in_pxl + taut_U2y_t2) ;
			}

			ap_int<FLOW_BW> p11_out_t1 = fxdpt_div<FLOW_BW, FLOW_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS,FLOW_FBITS,11>(taut_U1x_t3, ng1);
			ap_int<FLOW_BW> p12_out_t1 = fxdpt_div<FLOW_BW, FLOW_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS,FLOW_FBITS,11>(taut_U1y_t3, ng1);
			ap_int<FLOW_BW> p21_out_t1 = fxdpt_div<FLOW_BW, FLOW_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS,FLOW_FBITS,11>(taut_U2x_t3, ng2);
			ap_int<FLOW_BW> p22_out_t1 = fxdpt_div<FLOW_BW, FLOW_BW, FLOW_BW, FLOW_FBITS, FLOW_FBITS,FLOW_FBITS,11>(taut_U2y_t3, ng2);

			pxl_pack_out_P11_out.range(bit_out + FLOW_BW - 1, bit_out) = p11_out_t1;
			pxl_pack_out_P12_out.range(bit_out + FLOW_BW - 1, bit_out) = p12_out_t1;
			pxl_pack_out_P21_out.range(bit_out + FLOW_BW - 1, bit_out) = p21_out_t1;
			pxl_pack_out_P22_out.range(bit_out + FLOW_BW - 1, bit_out) = p22_out_t1;

			
		}
		// writing to output stream
		P11_out.write(wr_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_P11_out);    
		P12_out.write(wr_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_P12_out); 
		P21_out.write(wr_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_P21_out); 
		P22_out.write(wr_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_P22_out); 
		wr_ptr++;

	}

	return 0;
}//estimateDualVariables

_DUPMAT2_TPLT_DEC int dupMat(
		xf::cv::Mat<TYPE, ROWS, COLS, NPC>& Min,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH1>& Mout1,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH2>& Mout2,
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		WORDWIDTH_SRC = XF_WORDWIDTH(TYPE, NPC),
	};

	int loopbound = height*width;

	RowColLoop:
	for (int idx = 0; idx < loopbound; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*COLS/NPC max=ROWS*COLS/NPC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		XF_SNAME(WORDWIDTH_SRC) pxl_pack_in_1 = Min.read(idx);

		// writing into output stream
		Mout1.write(idx, pxl_pack_in_1); 
		Mout2.write(idx, pxl_pack_in_1); 

	}
	return 0;
}//dupMat

_DUPMAT4_TPLT_DEC int dupMat(
		xf::cv::Mat<TYPE, ROWS, COLS, NPC>& Min,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH1>& Mout1,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH2>& Mout2,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH3>& Mout3,
		xf::cv::Mat<TYPE, ROWS, COLS, NPC, DEPTH4>& Mout4,
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on

	enum
	{
		WORDWIDTH_SRC = XF_WORDWIDTH(TYPE, NPC),
	};
	int loopbound = height*width;

	RowColLoop:
	for (int idx = 0; idx < loopbound; idx++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*COLS/NPC max=ROWS*COLS/NPC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		XF_SNAME(WORDWIDTH_SRC) pxl_pack_in_1 = Min.read(idx);

		// writing into output stream
		Mout1.write(idx, pxl_pack_in_1); 
		Mout2.write(idx, pxl_pack_in_1); 
		Mout3.write(idx, pxl_pack_in_1); 
		Mout4.write(idx, pxl_pack_in_1); 
	}
	return 0;
}//dupMat

_MULFUNCT_TPLT_DEC int multiplyU(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_in,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_in,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1_out,	
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2_out, 	
		ap_int<SCALE_BW> scale,
		unsigned short height,
		unsigned short width_ncpr) {
		// clang-format off
#pragma HLS inline off
		// clang-format on

	enum
	{
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) ,
		INT_BW = 2*FLOW_BW
	};

	XF_SNAME(WORDWIDTH_DST) pxl_pack_out_U1, pxl_pack_out_U2;
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_U1, pxl_pack_in_U2;

	int read_ptr = 0 ;
	int write_ptr = 0 ;

	RowLoop:
	for (short i = 0; i < height; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		// clang-format on
		ColLoop:
		for (short j = 0; j < width_ncpr; j++) {
			// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline II=1
			// clang-format on

			//Read input stream
			pxl_pack_in_U1   = (XF_SNAME(WORDWIDTH_SRC2))(  U1_in.read(read_ptr));
			pxl_pack_in_U2   = (XF_SNAME(WORDWIDTH_SRC2))(  U2_in.read(read_ptr));
			read_ptr++;

			ProcLoop:
			for (int bit_in =0 , bit_out = 0; bit_in < ((FLOW_BW << XF_BITSHIFT(NPC)) ); bit_in += FLOW_BW ,bit_out += FLOW_BW) {
		// clang-format off
#pragma HLS unroll
		// clang-format on
				//Extract pixels
				XF_PTNAME(DEPTH_SRC2) U1_pxl 	= pxl_pack_in_U1.range(  bit_in + FLOW_BW- 1, bit_in);
				XF_PTNAME(DEPTH_SRC2) U2_pxl 	= pxl_pack_in_U2.range(  bit_in + FLOW_BW- 1, bit_in);

				//*********  Computation:
				//*********  out = in * scale
				ap_int<INT_BW> U1s = U1_pxl * scale; 	
				ap_int<INT_BW> U2s = U2_pxl * scale; 	

				//round & saturation
				ap_int<FLOW_BW> U1s_out = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS>(U1s);
				ap_int<FLOW_BW> U2s_out = rounding_n_saturation<INT_BW, FLOW_BW, FLOW_FBITS*2, FLOW_FBITS>(U2s);
				pxl_pack_out_U1.range(bit_out + FLOW_BW - 1, bit_out) = U1s_out;
				pxl_pack_out_U2.range(bit_out + FLOW_BW - 1, bit_out) = U2s_out;

			}
			// writing into output stream
			U1_out.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_U1); 
			U2_out.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out_U2); 
			write_ptr++;
		}
	}
	return 0;
}//multiplyU


template <int FLOW_T, int OUT_T, int ROWS, int COLS, int NPC, int FLOW_FBITS>
int mergeU(
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U1,
		xf::cv::Mat<FLOW_T, ROWS, COLS, NPC>& U2,
		xf::cv::Mat<OUT_T, ROWS, COLS, NPC>& U12,
		unsigned short height,
		unsigned short width) {
		// clang-format off
#pragma HLS inline off
		// clang-format on
	enum
	{
		PLANES = XF_CHANNELS(FLOW_T, NPC),
		DEPTH_SRC2 = XF_DEPTH(FLOW_T, NPC),
		DEPTH_DST  = XF_DEPTH(FLOW_T, NPC),
		WORDWIDTH_SRC2 = XF_WORDWIDTH(FLOW_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(FLOW_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC) / PLANES,
		INT_BW = 2*FLOW_BW
	};

	ap_uint<64*NPC> pxl_pack_out;
	XF_SNAME(WORDWIDTH_SRC2) pxl_pack_in_U1, pxl_pack_in_U2;

	int read_ptr = 0 ;
	int write_ptr = 0 ;

	RowColLoop:
	for (int i = 0; i < height*width; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS*TC max=ROWS*TC
#pragma HLS pipeline II=1
		// clang-format on

		//Read input stream
		pxl_pack_in_U1   = (XF_SNAME(WORDWIDTH_SRC2))(  U1.read(read_ptr));
		pxl_pack_in_U2   = (XF_SNAME(WORDWIDTH_SRC2))(  U2.read(read_ptr));
		read_ptr++;

		ProcLoop:
		for (int n = 0, bit_in = 0, bit_out = 0 ; n < NPC; n++, bit_in += FLOW_BW,  bit_out += 64) {
		// clang-format off
#pragma HLS unroll
		// clang-format on
			//Extract pixels
			XF_PTNAME(DEPTH_SRC2) U1_pxl 	= pxl_pack_in_U1.range(  bit_in + FLOW_BW- 1, bit_in);
			XF_PTNAME(DEPTH_SRC2) U2_pxl 	= pxl_pack_in_U2.range(  bit_in + FLOW_BW- 1, bit_in);

			float U1_val = (float)U1_pxl / (1<<FLOW_FBITS);
			float U2_val = (float)U2_pxl / (1<<FLOW_FBITS);

			ap_uint<32> *U1_int = (ap_uint<32> *)(&U1_val);
			ap_uint<32> *U2_int = (ap_uint<32> *)(&U2_val);

			ap_uint<64> merge;
			merge.range(31,0) = *U1_int;
			merge.range(63,32) = *U2_int;

			pxl_pack_out.range(bit_out + 64 - 1, bit_out) = merge;

		}
		// writing into output stream
		U12.write(write_ptr++, pxl_pack_out); 

	}
	return 0;
}//mergeU

template <int IN_T, int OUT_T, int ROWS, int COLS, int NPC, int IN_FBITS, int OUT_FBITS>
int ConvertType(
		xf::cv::Mat<IN_T, ROWS, COLS, NPC>& in,	
		xf::cv::Mat<OUT_T, ROWS, COLS, NPC>& out,
		unsigned short height,
		unsigned short width_ncpr) {
		// clang-format off
#pragma HLS inline off
		// clang-format on

	enum
	{
		DEPTH_SRC = XF_DEPTH(IN_T, NPC),
		DEPTH_DST  = XF_DEPTH(OUT_T, NPC),
		WORDWIDTH_SRC = XF_WORDWIDTH(IN_T, NPC),
		WORDWIDTH_DST  = XF_WORDWIDTH(OUT_T, NPC),
		TC = COLS >> XF_BITSHIFT(NPC),
		IN_BW = XF_PIXELWIDTH(IN_T, NPC) ,
		OUT_BW = XF_PIXELWIDTH(OUT_T, NPC) 
	};

	XF_SNAME(WORDWIDTH_DST) pxl_pack_out;
	XF_SNAME(WORDWIDTH_SRC) pxl_pack_in;

	int read_ptr = 0 ;
	int write_ptr = 0 ;
	RowLoop:
	for (short i = 0; i < height; i++) {
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		// clang-format on
		ColLoop:
		for (short j = 0; j < width_ncpr; j++) {
			// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline II=1
			// clang-format on

			//Read input stream
			pxl_pack_in   = (XF_SNAME(WORDWIDTH_SRC))(  in.read(read_ptr));
			read_ptr++;

			ProcLoop:
			for (int npc_cnt = 0, bit_in =0 , bit_out = 0; npc_cnt<NPC; npc_cnt++, bit_in += IN_BW ,bit_out += OUT_BW) {
		// clang-format off
#pragma HLS unroll
		// clang-format on
				//Extract pixels
				XF_PTNAME(DEPTH_SRC) in_pxl 	= pxl_pack_in.range(  bit_in + IN_BW - 1, bit_in);
				XF_PTNAME(DEPTH_DST) out_pxl;
				if(IN_FBITS>OUT_FBITS)
					out_pxl =((XF_PTNAME(DEPTH_DST))in_pxl)>>(IN_FBITS-OUT_FBITS);
				else
					out_pxl =((XF_PTNAME(DEPTH_DST))in_pxl)<<(OUT_FBITS-IN_FBITS);
				 
				pxl_pack_out.range(bit_out + OUT_BW - 1, bit_out) = out_pxl;
			}
			// writing into output stream
			out.write(write_ptr, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out); 
			write_ptr++;
		}
	}
	return 0;
}//ConvertType

template <int FLOW_PTRWIDTH = 16, int FLOW_T, int NPC>
void initU(
		ap_uint<FLOW_PTRWIDTH>* _U1,
		ap_uint<FLOW_PTRWIDTH>* _U2,
		int height,
		int width)
{
		// clang-format off
#pragma HLS inline off
		// clang-format on

	enum {
		FLOW_BW = XF_PIXELWIDTH(FLOW_T, NPC)
	};

	int loop_bound = (height*width + (FLOW_PTRWIDTH/FLOW_BW) - 1)/(FLOW_PTRWIDTH/FLOW_BW);

	for(int idx = 0; idx < loop_bound; idx++)
	{
		// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=52*91 max=52*91
#pragma HLS pipeline II=1
		// clang-format on

		_U1[idx] = 0;
		_U2[idx] = 0;

	}

}//initU

// ======================================================================================

// Some clean up for macros used
#undef _GENERIC_REMAP_TPLT_DEC
#undef _GENERIC_REMAP_TPLT
#undef _GENERIC_REMAP

#undef CH_IDX_T
#undef K_ROW_IDX_T
#undef K_COL_IDX_T
#undef COL_IDX_T
#undef ROW_IDX_T
#undef SIZE_IDX_T
#undef FLOW_VAL_T
#undef FLOW_BITWIDTH
#undef FLOW_FBITS
#undef BICUBIC_FILTER

#undef _NPPC
#undef _NPPC_SHIFT_VAL
#undef _ECPR
#undef _NP_IN_PREV
#undef _DST_PIX_WIDTH

} // namespace cv
} // namespace xf
