#ifndef __XF_EXTRA_UTILITY_HPP__
#define __XF_EXTRA_UTILITY_HPP__

#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"
#include "ap_axi_sdata.h"

namespace xf {
  namespace cv {
/*
    // ======================================================================================
    // Function to split hls::stream into 2 hls::stream
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void DuplicateStrm(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                       hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                       hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                       uint16_t img_height, uint16_t img_width) {

        img_width = img_width >> XF_BITSHIFT(NPC);
        ap_uint<13> row, col;

        Row_Loop:
        for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
            Col_Loop:
            for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
                XF_TNAME(SRC_T, NPC) tmp_src;
                tmp_src = _src_mat.read();
                _dst1_mat.write(tmp_src);
                _dst2_mat.write(tmp_src);
            }
        }
    } // End of DuplicateStrm()
    // ======================================================================================

    // ======================================================================================
    // Function to split hls::stream into 3 hls::stream
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void DuplicateStrm_3(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst3_mat,
                         uint16_t img_height, uint16_t img_width) {

        img_width = img_width >> XF_BITSHIFT(NPC);
        ap_uint<13> row, col;

        Row_Loop:
        for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
            Col_Loop:
            for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
                XF_TNAME(SRC_T, NPC) tmp_src;
                tmp_src = _src_mat.read();
                _dst1_mat.write(tmp_src);
                _dst2_mat.write(tmp_src);
                _dst3_mat.write(tmp_src);
            }
        }
    } // End of DuplicateStrm_3()
    // ======================================================================================

    // ======================================================================================
    // Function to split hls::stream into 4 hls::stream
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void DuplicateStrm_4(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst3_mat,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& _dst4_mat,
                         uint16_t img_height, uint16_t img_width) {

        img_width = img_width >> XF_BITSHIFT(NPC);
        ap_uint<13> row, col;

        Row_Loop:
        for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
            Col_Loop:
            for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
                XF_TNAME(SRC_T, NPC) tmp_src;
                tmp_src = _src_mat.read();
                _dst1_mat.write(tmp_src);
                _dst2_mat.write(tmp_src);
                _dst3_mat.write(tmp_src);
                _dst4_mat.write(tmp_src);
            }
        }
    } // End of DuplicateStrm_4()
    // ======================================================================================


    // ======================================================================================
    // Function to split xf::cv::Mat into 2 xf::cv::Mat
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void duplicateMat(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                      xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                      xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2) {
#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

        int _rows = _src.rows;
        int _cols = _src.cols;

        hls::stream<XF_TNAME(SRC_T, NPC)> src;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst1;

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
            }
        }

        DuplicateStrm<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, _rows, _cols);

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
                _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
            }
        }
    } // End of duplicateMat()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 3 xf::cv::Mat
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void duplicateMat_3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst3) {
#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

        int _rows = _src.rows;
        int _cols = _src.cols;

        hls::stream<XF_TNAME(SRC_T, NPC)> src;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst1;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst2;

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
            }
        }

        DuplicateStrm_3<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, dst2, _rows, _cols);

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
                _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
                _dst3.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst2.read());
            }
        }
    } // End of duplicateMat_3()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 4 xf::cv::Mat
    // ======================================================================================
    template <int SRC_T, int ROWS, int COLS, int NPC>
    void duplicateMat_4(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst3,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst4) {
#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

        int _rows = _src.rows;
        int _cols = _src.cols;

        hls::stream<XF_TNAME(SRC_T, NPC)> src;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst1;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst2;
        hls::stream<XF_TNAME(SRC_T, NPC)> dst3;

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
            }
        }

        DuplicateStrm_4<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, dst2, dst3, _rows, _cols);

        for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
                _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
                _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
                _dst3.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst2.read());
                _dst4.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst3.read());
            }
        }
    } // End of duplicateMat_4()
    // ======================================================================================
*/
    // ======================================================================================
    // Function to read from DDR and copy to xf::cv::Mat
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void Ptr2xfMat(ap_uint<BUS_WIDTH> *in_ptr,
                   xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = out_mat.rows * (out_mat.cols >> XF_BITSHIFT(NPPC));
      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        out_mat.write(i, (XF_TNAME(TYPE, NPPC))in_ptr[i]);
      }

    } // End of xFDuplicateMat_PTRMAT()
	
	    // ======================================================================================
    // Function to read from DDR and copy to xf::cv::Mat
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xfMat2Ptr(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
					ap_uint<BUS_WIDTH> *out_ptr) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        out_ptr[i] = in_mat.read(i);
      }

    } // End of xFDuplicateMat_PTRMAT()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 2 streams (1 for DDR PTR and 1 for xf::cv::Mat)
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_PTRMAT(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                               ap_uint<BUS_WIDTH> *out_ptr,
                               xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);

        out_ptr[i] = (ap_uint<BUS_WIDTH>)tmp;
        out_mat.write(i, tmp);
      }

    } // End of xFDuplicateMat_PTRMAT()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 3 streams (1 for DDR PTR and 2 for xf::cv::Mat)
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_PTRMAT2(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                               ap_uint<BUS_WIDTH> *out_ptr,
                               xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat1,
                               xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat2) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);

        out_ptr[i] = (ap_uint<BUS_WIDTH>)tmp;
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
        //out_mat2.write(i, (XF_TNAME(XF_16SC1, NPPC))tmp); // TODO: Remove me as I am for experiment
      }

    } // End of xFDuplicateMat_PTRMAT2()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 3 streams (1 for DDR PTR, 1 for xf::cv::Mat and 1 for AXI stream)
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_PTR_MAT_AXI(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                                    ap_uint<BUS_WIDTH> *out_ptr,
                                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat,
                                    hls::stream<ap_axiu<BUS_WIDTH, 0, 0, 0> > &out_axi) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        ap_axiu<BUS_WIDTH, 0, 0, 0> v;
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);

        out_ptr[i] = tmp;
        out_mat.write(i, tmp);

        v.data = tmp;
        out_axi.write(v);
      }

    } // End of xFDuplicateMat_PTR_MAT_AXI()
    // ======================================================================================

    // ======================================================================================
    // Function to stream out xf::cv::Mat on AXI bus for K2K streaming
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xFMat2AXI_Strm(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                        hls::stream<ap_axiu<BUS_WIDTH, 0, 0, 0> > &out_axi) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        ap_axiu<BUS_WIDTH, 0, 0, 0> v;

        v.data = in_mat.read(i);
        out_axi.write(v);
      }

    } // End of xFDuplicateMat_PTR_AXI()
    // ======================================================================================

    // ======================================================================================
    // Function to read AXI stream into xf::cv::Mat for K2K streaming
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void AXI_Strm2xFMat(hls::stream<ap_axiu<BUS_WIDTH, 0, 0, 0> > &in_axi,
                        xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = out_mat.rows * (out_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT
        ap_axiu<BUS_WIDTH, 0, 0, 0> v = in_axi.read();

        out_mat.write(i, v.data);
      }

    } // End of xFDuplicateMat_PTR_AXI()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 2 streams (1 for DDR PTR and 1 for AXI stream)
    // ======================================================================================
    template<int BUS_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_PTR_AXI(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                                ap_uint<BUS_WIDTH> *out_ptr,
                                hls::stream<ap_axiu<BUS_WIDTH, 0, 0, 0> > &out_axi) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT
        ap_axiu<BUS_WIDTH, 0, 0, 0> v;
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);

        out_ptr[i] = tmp;

        v.data = tmp;
        out_axi.write(v);
      }

    } // End of xFDuplicateMat_PTR_AXI()
    // ======================================================================================


    // ======================================================================================
    // Function to split xf::cv::Mat into 2 xf::cv::Mat
    // ======================================================================================
    template<int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                        xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat1,
                        xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat2) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
      }

    } // End of xFDuplicateMat()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 3 xf::cv::Mat
    // ======================================================================================
    template<int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_3(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat1,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat2,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat3) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
        out_mat3.write(i, tmp);
      }

    } // End of xFDuplicateMat_3()
    // ======================================================================================

    // ======================================================================================
    // Function to split xf::cv::Mat into 4 xf::cv::Mat
    // ======================================================================================
    template<int TYPE, int ROWS, int COLS, int NPPC>
    void xFDuplicateMat_4(xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &in_mat,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat1,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat2,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat3,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPPC> &out_mat4) {
      #pragma HLS INLINE OFF

      const int c_TRIP_COUNT = ROWS*COLS;
      int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

      for (int i=0; i < loopcount; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS LOOP_TRIPCOUNT min=c_TRIP_COUNT max=c_TRIP_COUNT

        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
        out_mat3.write(i, tmp);
        out_mat4.write(i, tmp);
      }

    } // End of xFDuplicateMat_4()
    // ======================================================================================

    // ======================================================================================
    // Function to set border in the extracted kernel sized block
    // ======================================================================================
    template<int K_ROWS, int K_COLS, typename SRC_T, int BORDER_T>
    void xFSetBorder(xf::cv::Window<K_ROWS, K_COLS, SRC_T> &src_blk,
                    uint16_t _row, uint16_t _col, uint16_t _src_rows, uint16_t _src_cols) {
       #pragma HLS INLINE OFF

       uint16_t blk_t_idx, blk_b_idx;
       uint16_t blk_l_idx, blk_r_idx;

       blk_t_idx = (K_ROWS - _row - 1);
       blk_b_idx = (K_ROWS - (_row - _src_rows + 1) - 1);

       blk_l_idx = (K_COLS - _col - 1);
       blk_r_idx = (K_COLS - (_col - _src_cols + 1) - 1);

       for (uint16_t r = 0; r < K_ROWS; r++) {
         #pragma HLS UNROLL
         for (uint16_t c = 0; c < K_COLS; c++) {
           #pragma HLS UNROLL

           bool top_border    = ((r < blk_t_idx) && (_row < K_ROWS-1))   ? true : false;
           bool bottom_border = ((r > blk_b_idx) && (_row >= _src_rows)) ? true : false;
           bool left_border   = ((c < blk_l_idx) && (_col < K_COLS-1))   ? true : false;
           bool right_border  = ((c > blk_r_idx) && (_col >= _src_cols)) ? true : false;

           uint16_t r_idx = r, c_idx = c;

           if (BORDER_T == XF_BORDER_REPLICATE) {
             r_idx = top_border ? blk_t_idx : bottom_border ? blk_b_idx : r;

           } else if (BORDER_T == XF_BORDER_REFLECT_101) {
             r_idx = top_border ? (2*blk_t_idx - r) : bottom_border ? (2*blk_b_idx - r) : r;

           } else if (BORDER_T == XF_BORDER_REFLECT) {
             r_idx = top_border ? (2*blk_t_idx - r - 1) : bottom_border ? (2*blk_b_idx - r + 1) : r;

           } else { // TODO: Need to add other modes support
             r_idx = r;
           }

           if (BORDER_T == XF_BORDER_REPLICATE) {
             c_idx = left_border ? blk_l_idx : right_border ? blk_r_idx : c;

           } else if (BORDER_T == XF_BORDER_REFLECT_101) {
             c_idx = left_border ? (2*blk_l_idx - c) : right_border ? (2*blk_r_idx - c) : c;

           } else if (BORDER_T == XF_BORDER_REFLECT) {
             c_idx = left_border ? (2*blk_l_idx - c - 1) : right_border ? (2*blk_r_idx - c + 1) : c;

           } else { // TODO: Need to add other modes support
             c_idx = c;
           }


           if ((top_border | bottom_border | left_border | right_border) && (BORDER_T == XF_BORDER_CONSTANT)) {
             src_blk.val[r][c] = 0;
           } else {
             src_blk.val[r][c] = src_blk.val[r_idx][c_idx];
           }

         }
       }

    } // End of xFSetBorder()
    // ======================================================================================

  } // namespace cv
} // namespace xf

#endif //__XF_EXTRA_UTILITY_HPP__
