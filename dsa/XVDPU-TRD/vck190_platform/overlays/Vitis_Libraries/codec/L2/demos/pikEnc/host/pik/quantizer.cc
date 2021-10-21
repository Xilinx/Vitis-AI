// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "ap_int.h"
#include "pik/ac_strategy.h"
#include "pik/quantizer.h"
#include <algorithm>
#include <sstream>
#include <stdio.h>
#include <vector>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/arch_specific.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/profiler.h"
#include "pik/quant_weights.h"
#include "pik/simd/simd.h"

namespace pik {

static const int kDefaultQuant = 64;

Quantizer::Quantizer(const DequantMatrices *dequant, int quant_xsize,
                     int quant_ysize)
    : Quantizer(dequant, quant_xsize, quant_ysize, kDefaultQuant,
                kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(const DequantMatrices *dequant, int quant_xsize,
                     int quant_ysize, int quant_dc, int global_scale)
    : quant_xsize_(quant_xsize), quant_ysize_(quant_ysize),
      global_scale_(global_scale), quant_dc_(quant_dc),
      quant_img_ac_(quant_xsize_, quant_ysize_), dequant_(dequant) {
  RecomputeFromGlobalScale();

  FillImage(kDefaultQuant, &quant_img_ac_);

  memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
}

// TODO(veluca): reclaim the unused bit in global_scale encoding.
std::string Quantizer::Encode(PikImageSizeInfo *info) const {
  std::stringstream ss;
  int global_scale = global_scale_ - 1;
  ss << std::string(1, global_scale >> 8);
  ss << std::string(1, global_scale & 0xff);
  ss << std::string(1, quant_dc_ - 1);

  std::cout<<std::dec<<"global_scale_="<<global_scale_<<std::endl;
  std::cout<<std::dec<<"quant_dc_="<<quant_dc_<<std::endl;
  if (info) {
    info->total_size += 3;
  }
  return ss.str();
}

bool Quantizer::Decode(BitReader *br) {
  int global_scale = br->ReadBits(8) << 8;
  global_scale |= br->ReadBits(8);
  global_scale_ = (global_scale & 0x7FFF) + 1;
  quant_dc_ = br->ReadBits(8) + 1;
  RecomputeFromGlobalScale();
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;
  return true;
}

void Quantizer::DumpQuantizationMap() const {
  printf("Global scale: %d (%.7f)\nDC quant: %d\n", global_scale_,
         global_scale_ * 1.0 / kGlobalScaleDenom, quant_dc_);
  printf("AC quantization Map:\n");
  for (size_t y = 0; y < quant_img_ac_.ysize(); ++y) {
    for (size_t x = 0; x < quant_img_ac_.xsize(); ++x) {
      printf(" %3d", quant_img_ac_.Row(y)[x]);
    }
    printf("\n");
  }
}

// Works in "DC image", i.e. transforms every pixel.
Image3S QuantizeCoeffsDC(const Image3F &dc, const Quantizer &quantizer) {
  const size_t xsize_blocks = dc.xsize();
  const size_t ysize_blocks = dc.ysize();
  Image3S out(xsize_blocks, ysize_blocks);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float *PIK_RESTRICT row_in = dc.PlaneRow(c, by);
      int16_t *PIK_RESTRICT row_out = out.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        row_out[bx] = quantizer.QuantizeDC(c, row_in[bx]);
      }
    }
  }

  return out;
}

ImageF QuantizeRoundtripDC(const Quantizer &quantizer, int c,
                           const ImageF &dc) {
  // All coordinates are blocks.
  const int xsize_blocks = dc.xsize();
  const int ysize_blocks = dc.ysize();
  ImageF out(xsize_blocks, ysize_blocks);

  // Always use DCT8 quantization kind for DC
  const float mul = quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                    quantizer.inv_quant_dc();

  for (size_t by = 0; by < ysize_blocks; ++by) {
    const float *PIK_RESTRICT row_in = dc.ConstRow(by);
    float *PIK_RESTRICT row_out = out.Row(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      row_out[bx] = quantizer.QuantizeDC(c, row_in[bx]) * mul;
    }
  }
  return out;
}
/*
#define SHIFT_SCL  (15)
#define FACTOR_SCL  (1<<SHIFT_SCL)
#define SCLF( a) ((int)(a*FACTOR_SCL))
typedef  short ap_frac;
typedef  unsigned short apu_frac;
typedef  short ap_frac_1;
typedef  int   ap_frac_15;
*/
#define SHIFT_SCL (6)
#define FACTOR_SCL (1 << SHIFT_SCL)
#define SCLF(a) ((int)(a * FACTOR_SCL))
typedef ap_int<SHIFT_SCL + 1> ap_frac;
typedef ap_uint<SHIFT_SCL + 1> apu_frac;
typedef ap_int<SHIFT_SCL + 1> ap_frac_1;
typedef ap_int<SHIFT_SCL + 10> ap_frac_15;

int16_t UpdateErr_int(ap_frac_15 val_i, apu_frac thres_i,
                      char k //, int previous_row_err_i[8]
                      ) {
#pragma HLS inline
  static ap_frac err_left;
  static ap_frac previous_row_err_i[8];
#pragma HLS ARRAY_PARTITION variable = previous_row_err_i complete dim = 1
  int idx = k & 7;
  short err_i;

  if (k == 0)
    err_i = 0;
  else if ((idx) == 0) {
    err_i = previous_row_err_i[idx];
  } else {
    if (k > 7)
      err_i = err_left + previous_row_err_i[idx];
    else
      err_i = err_left;
  }
  bool isPos = val_i > 0;

  int val_org_i = val_i;
  bool isValOrg_1 = (val_org_i > FACTOR_SCL) || (0 - val_org_i > FACTOR_SCL);
  apu_frac val_frac = val_i & (FACTOR_SCL - 1);
  ap_frac_15 val_int = (val_i - val_frac) >> SHIFT_SCL;
  bool isValIntZero = val_int == 0;
  bool isValNegOne = val_int == -1;

  bool isZero_u;
  bool isZero_Nu;
  bool isUseErr = (err_i > 0);
  ap_frac_1 gap_u_Z_p = (0 << SHIFT_SCL) + val_frac + err_i / 2;       //;
  ap_frac_1 gap_u_Z_n = -(((-1) << SHIFT_SCL) + val_frac - err_i / 2); //;
  ap_frac_1 gap_u_Nz_p =
      val_frac + err_i / 2 -
      (((val_frac + err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL); //;
  ap_frac_1 gap_u_Nz_n =
      (((val_frac - err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL) -
      val_frac + err_i / 2; //;
  ap_frac_1 gap_un_Z_p = val_frac;
  ap_frac_1 gap_un_Z_n = (FACTOR_SCL - val_frac); //-val_i;;
  ap_frac_1 gap_un_Nz_p =
      val_frac - (((val_frac + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL);
  ;
  ap_frac_1 gap_un_Nz_n =
      (((val_frac + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL) - val_frac;
  ;
  ap_frac_1 err_i_gap = isValOrg_1 ? 0 : err_i;
  ap_frac_1 err_i_gap_un_Z_p = err_i_gap + gap_un_Z_p;
  ap_frac_1 err_i_gap_un_Z_n = err_i_gap + gap_un_Z_n;
  ap_frac_1 err_i_gap_un_Nz_p = err_i_gap + gap_un_Nz_p;
  ap_frac_1 err_i_gap_un_Nz_n = err_i_gap + gap_un_Nz_n;
  bool NoCarry_u_p = val_frac < (thres_i - err_i / 2);
  bool NoCarry_u_n = (FACTOR_SCL - val_frac) < (thres_i - err_i / 2);
  bool NoCarry_Nu_p = val_frac < thres_i;
  bool NoCarry_Nu_n = (FACTOR_SCL - val_frac) < thres_i;

  if (isUseErr) {
    if (isPos) {
      if (isValIntZero && NoCarry_u_p)
        isZero_u = true;
      else
        isZero_u = false;
    } else {
      if (isValNegOne && NoCarry_u_n)
        isZero_u = true;
      else
        isZero_u = false;
    }
  }
  if (!isUseErr) {
    if (isPos) {
      if (isValIntZero && NoCarry_Nu_p)
        isZero_Nu = true;
      else
        isZero_Nu = false;
    } else {
      if (isValNegOne && NoCarry_Nu_n)
        isZero_Nu = true;
      else
        isZero_Nu = false;
    }
  }

  if (k == 0 || (idx) == 7) {
    err_left = 0;
  } else if (isUseErr) {
    if (isZero_u) {
      if (isPos)
        err_left = gap_u_Z_p / 2;
      else
        err_left = gap_u_Z_n / 2;
    } else {
      if (isPos)
        err_left = gap_u_Nz_p / 2;
      else
        err_left = gap_u_Nz_n / 2;
    }
  } else {
    if (isZero_Nu) {
      if (isPos)
        err_left = err_i_gap_un_Z_p / 2; // + gap_un_Z_p;
      else
        err_left = err_i_gap_un_Z_n / 2; // + gap_un_Z_n;
    } else {
      if (isPos)
        err_left = err_i_gap_un_Nz_p / 2; // + gap_un_Nz_p;
      else
        err_left = err_i_gap_un_Nz_n / 2; // + gap_un_Nz_n;
    }
  }

  ap_frac err_new_i;
  if (k == 0) {
    err_new_i = 0;
  } else if (isUseErr) {
    if (isZero_u) {
      if (isPos)
        err_new_i = gap_u_Z_p;
      else
        err_new_i = gap_u_Z_n;
    } else {
      if (isPos)
        err_new_i = gap_u_Nz_p;
      else
        err_new_i = gap_u_Nz_n;
    }
  } else {
    if (isZero_Nu) {
      if (isPos)
        err_new_i = err_i_gap_un_Z_p; // + gap_un_Z_p;
      else
        err_new_i = err_i_gap_un_Z_n; // + gap_un_Z_n;
    } else {
      if (isPos)
        err_new_i = err_i_gap_un_Nz_p; // + gap_un_Nz_p;
      else
        err_new_i = err_i_gap_un_Nz_n; // + gap_un_Nz_n;
    }
  }

  if ((idx) == 7)
    previous_row_err_i[idx] = err_new_i; // 1.0 * err;
  else
    previous_row_err_i[idx] = err_new_i / 2;

  int16_t v_i;
  if (isUseErr) {
    if (isPos)
      v_i = isZero_u ? 0 : (int16_t)(((val_i + err_i / 2 + FACTOR_SCL / 2) >>
                                      SHIFT_SCL));
    else
      v_i = isZero_u ? 0 : (int16_t)(((val_i - err_i / 2 + FACTOR_SCL / 2) >>
                                      SHIFT_SCL));
  } else // err is not used
    v_i = isZero_Nu ? 0 : (int16_t)(((val_i + FACTOR_SCL / 2) >> SHIFT_SCL));
  if (v_i > 32767)
    v_i = 32767;
  if (v_i < -32767)
    v_i = -32767;

  return v_i;
}

void QuantizeBlockAC_core_L1(
    uint8_t quant_table, int32_t quant, size_t quant_kind, int c, size_t xsize,
    size_t ysize, const float *block_in, size_t in_stride, int16_t *block_out,
    size_t out_stride,
    const float *qm,   //    = dequant_->InvMatrix(quant_table, quant_kind, c);
    const float qac,   // = Scale() * quant;
    const float thres, // = zero_bias_[c];
    size_t block_shift) {

  size_t kBlockSize = kBlockDim * kBlockDim;
  //std::cout<<"instride="<<in_stride<<std::endl;
  for (size_t iy = 0; iy < ysize; iy++) {
    for (size_t ix = 0; ix < xsize; ix++) {
      for (char k = 0; k < kBlockSize; ++k) {
#pragma HLS pipeline II = 1
        size_t x = xsize * (k % kBlockDim) + ix;
        size_t y = ysize * (k / kBlockDim) + iy;
        size_t pos = y * kBlockDim * xsize + x;
        size_t block_off = pos >> block_shift;
        size_t block_idx = pos & (xsize * kBlockDim * kBlockDim - 1);
        float v_block_in = block_in[block_off * in_stride + block_idx]; //
        float v_qm = qm[pos];
        float v_qac = qac;
        float val = v_block_in * (v_qm * v_qac);
        ap_frac_15 val_i = SCLF(val);
        apu_frac thres_i = SCLF(thres);
        int16_t v_i = UpdateErr_int(val_i, thres_i, k); //, previous_row_err_i);
        block_out[block_off * out_stride + block_idx] = v_i;

        std::cout<<"std_qua: iy="<<iy<<" ix="<<ix<<" k="<<(int)k<<" cplane="
        		 <<v_block_in<<" val="<<val<<" by=" <<block_off<<" bx="<<block_idx
				 <<" qm="<<v_qm<<" qac="<<v_qac
				 <<std::dec<<" quantized="<<v_i<<std::endl;
        /*

         std::cout << "k2_qua: iy=" << iy << " ix=" << ix << " k=" << k
        		  << " cplane=" << plane << " val=" << val
				  << " by=" << addr_o(9, 5) << " bx=" << addr_o(4, 0)
				  << " qm=" << qm << " qac=" << qac[iy][ix]
				  << " quantized=" << v << std::endl;

         */
      }
    }
  }
}

/*
void Quantizer::QuantizeBlockAC(uint8_t quant_table, int32_t quant,
                                size_t quant_kind, int c, size_t xsize,
                                size_t ysize, const float *block_in,
                                size_t in_stride, int16_t *block_out,
                                size_t out_stride) const {
  constexpr size_t kBlockSize = kBlockDim * kBlockDim;
  const float *qm = dequant_->InvMatrix(quant_table, quant_kind, c);
  const float qac = Scale() * quant;

  const float thres = zero_bias_[c];
  size_t block_shift =
      NumZeroBitsBelowLSBNonzero(kBlockDim * kBlockDim * xsize);
  //std::cout<<"block_shift="<<block_shift<<" out_stride="<<out_stride<<std::endl;
  QuantizeBlockAC_core_L1(quant_table, quant, quant_kind, c, xsize, ysize,
                          block_in, in_stride, block_out, out_stride, qm, qac,
                          thres, block_shift);
}
*/
} // namespace pik
