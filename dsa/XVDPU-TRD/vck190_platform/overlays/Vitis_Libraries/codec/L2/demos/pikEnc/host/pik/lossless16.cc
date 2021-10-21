// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#include "pik/lossless16.h"

#include <cmath>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pik/lossless_entropy.h"

namespace pik {

namespace {

static const int kGroupSize = 512,
                 kGroupSize2plus = kGroupSize * kGroupSize * 9 / 8,
    kWithSign1 = 4, kBitsMax1 = 13, kNumContexts1 = 1 + kWithSign1 + kBitsMax1,
    kWithSign3 = 3, kBitsMax3 = 13, kNumContexts3 = 1 + kWithSign3 + kBitsMax3,
    kMaxError = 0x3fbf,  kMaxSumErrors = (kMaxError + 1) * 4,  kNumRuns = 1;

// TODO(lode): split state variables needed for encoder from those for decoder
//             and run const init just once!  ~65536*3 'const' values in State()
struct State {
  int prediction0, prediction1, prediction2, prediction3,
      width, WithSIGN, BitsMAX, NumCONTEXTS;
  uint16_t *PIK_RESTRICT rowImg, *PIK_RESTRICT rowPrev;

  uint16_t edata[kNumContexts1 > kNumContexts3 ? kNumContexts1 : kNumContexts3]
                [kGroupSize * kGroupSize];
  uint8_t compressedDataTmpBuf[kGroupSize2plus], *compressedData;
  int32_t errors0[kGroupSize * 2];  // Errors of predictor 0
  int32_t errors1[kGroupSize * 2];  // Errors of predictor 1
  int32_t errors2[kGroupSize * 2];  // Errors of predictor 2
  int32_t errors3[kGroupSize * 2];  // Errors of predictor 3
  uint8_t nbitErr[kGroupSize * 2];
  int32_t trueErr[kGroupSize * 2];

  uint16_t error2weight[kMaxSumErrors],            // const
           sign_LSB_forward_transform[0x10000],    // const
           sign_LSB_backward_transform[0x10000];   // const
  uint8_t numBitsTable[256];                       // const

  State() {
    for (int i = 0; i < 256; ++i)
      numBitsTable[i] = numbitsInit(i);  // const init!

    error2weight[0] = 0xffff;
    for (int j = 1; j < kMaxSumErrors; ++j)
      error2weight[j] = 181 * 256 / j;   // const init!

    // For compress
    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_forward_transform[i] =
          (i & 32768 ? (0xffff - i) * 2 + 1 : i * 2);  // const init!

    // For decompress
    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_backward_transform[i] =
          (i & 1 ? 0xffff - (i >> 1) : i >> 1);  // const init!

    // Prevent uninitialized values in case of invalid compressed data
    memset(edata, 0, sizeof(edata));
  }

  PIK_INLINE int numbitsInit(int x) {
    assert(0 <= x && x <= 255);
    int res = 0;
    if (x >= 16) res = 4, x >>= 4;
    if (x >= 4) res += 2, x >>= 2;
    return (res + std::min(x, 2));
  }

  PIK_INLINE int numBits(int x) {
    assert(0 <= x && x <= 0xffff);
    if (x < 256) return numBitsTable[x];
    return std::min(8 + numBitsTable[x >> 8], BitsMAX);
  }

  PIK_INLINE int predict1y0(size_t x, size_t yp, size_t yp1, int& maxErr) {
    maxErr = (x == 0 ? NumCONTEXTS - 1
                     : x == 1 ? nbitErr[yp - 1]
                              : std::max(nbitErr[yp - 1], nbitErr[yp - 2]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 14 * 256  // 14
                : x == 1 ? rowImg[x - 1]
                         : rowImg[x - 1] + (rowImg[x - 1] - rowImg[x - 2]) / 4);
    return (prediction0 < 0 ? 0 : prediction0 > 0xffff ? 0xffff : prediction0);
  }

  PIK_INLINE int predict1x0(size_t x, size_t yp, size_t yp1, int& maxErr) {
    maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < width ? 1 : 0)]);
    prediction0 = prediction2 = prediction3 = rowPrev[x];
    prediction1 = (rowPrev[x] * 3 + rowPrev[x + (x < width ? 1 : 0)] + 2) >> 2;
    return prediction1;
  }

  PIK_INLINE int predict1(size_t x, size_t yp, size_t yp1, int& maxErr) {
    if (!rowPrev) return predict1y0(x, yp, yp1, maxErr);
    if (x == 0LL) return predict1x0(x, yp, yp1, maxErr);
    int weight0 = errors0[yp - 1] + errors0[yp1] + errors0[yp1 - 1];
    int weight1 = errors1[yp - 1] + errors1[yp1] + errors1[yp1 - 1];
    int weight2 = errors2[yp - 1] + errors2[yp1] + errors2[yp1 - 1];
    int weight3 = errors3[yp - 1] + errors3[yp1] + errors3[yp1 - 1];
    uint8_t mxe = nbitErr[yp - 1];
    mxe = std::max(mxe, nbitErr[yp1]);
    mxe = std::max(mxe, nbitErr[yp1 - 1]);
    int N = rowPrev[x], W = rowImg[x - 1],
        NE = N;  // NW = rowPrev[x - 1] unused!
    if (x < width) {
      mxe = std::max(mxe, nbitErr[yp1 + 1]), NE = rowPrev[x + 1];
      weight0 += errors0[yp1 + 1];
      weight1 += errors1[yp1 + 1];
      weight2 += errors2[yp1 + 1];
      weight3 += errors3[yp1 + 1];
    }

    weight0 = error2weight[weight0] + 1;
    weight1 = error2weight[weight1] + 1;
    weight2 = error2weight[weight2];
    weight3 = error2weight[weight3];

    int teW = trueErr[yp - 1];  // range: -0xffff...0xffff
    int teN = trueErr[yp1];
    int teNW = trueErr[yp1 - 1];
    int sumWN = teN + teW;  // range: -0x1fffe...0x1fffe
    int teNE = (x < width ? trueErr[yp1 + 1] : 0);

    prediction0 = N - sumWN * 3 / 4;                          // 24/32
    prediction1 = W - (sumWN + teNW) * 11 / 32;               // 11/32
    prediction2 = W + (((NE - N) * 13 + 7) >> 4);             // 26/32
    prediction3 = N - (((teN + teNW + teNE) * 7 + 29) >> 5);  //  7/32
    int sumWeights = weight0 + weight1 + weight2 + weight3;
    int64_t s = sumWeights * 3 / 8;
    s += ((int64_t)prediction0) * weight0;
    s += ((int64_t)prediction1) * weight1;
    s += ((int64_t)prediction2) * weight2;
    s += ((int64_t)prediction3) * weight3;
    int prediction = s / sumWeights;

    if (mxe && mxe <= WithSIGN * 2) {
      if (teW * 3 + teN * 2 + teNW + teNE < 0) --mxe;  // 3 2 1 1
    }
    maxErr = mxe;

    int mx = std::max(N - 28, std::max(W, NE));  // 28
    int mn = std::min(N + 28, std::min(W, NE));  // 28
    prediction = std::max(mn, std::min(mx, prediction));
    return prediction;
  }

  bool IsRLE(const uint8_t* data, size_t size) {
    if (size < 4) return false;
    uint8_t first = data[0];
    for (size_t i = 1; i < size; i++) {
      if (data[i] != first) return false;
    }
    return true;
  }

  // TODO(lode): move this to lossless_entropy.cc
  bool compressWithEntropyCode(size_t* pos, size_t S, uint8_t* compressedBuf) {
    if (S == 0) {
      *pos += encodeVarInt(0, &compressedBuf[*pos]);
      return true;
    }
    uint8_t* src = &compressedBuf[*pos + 8];
    size_t cs;
    if (IsRLE(src, S)) {
      cs = 1;  // use RLE encoding instead
    } else {
      if (!MaybeEntropyEncode(src, S, sizeof(compressedDataTmpBuf),
                              &compressedDataTmpBuf[0], &cs)) {
        return PIK_FAILURE("lossless16 entropy encode");
      }
    }
    if (cs >= S) cs = 0;  // EntropyCode worse than original, use memcpy.
    *pos += encodeVarInt(cs <= 1 ? (S - 1) * 3 + 1 + cs : cs * 3,
                         &compressedBuf[*pos]);
    uint8_t* dst = &compressedBuf[*pos];
    if (cs == 1)
      compressedBuf[(*pos)++] = *src;
    else if (cs == 0)
      memmove(dst, src, S), *pos += S;
    else
      memcpy(dst, &compressedDataTmpBuf[0], cs), *pos += cs;
    return true;
  }

  // TODO(lode): move this to lossless_entropy.cc
  // ds = decompressed size output
  bool decompressWithEntropyCode(uint8_t* dst, size_t dst_capacity,
                                 const uint8_t* src, size_t src_capacity,
                                 size_t* ds, size_t* pos) {
    size_t cs = decodeVarInt(src, src_capacity, pos);
    if (cs == 0) {
      *ds = 0;
      return true;
    }
    size_t mode = cs % 3;
    cs /= 3;
    if (mode == 2) {
      if (*pos >= src_capacity) return PIK_FAILURE("entropy decode failed");
      if (cs + 1 > dst_capacity) return PIK_FAILURE("entropy decode failed");
      memset(dst, src[(*pos)++], ++cs);
      *ds = cs;
    } else if (mode == 1) {
      if (*pos + cs + 1 > src_capacity)
        return PIK_FAILURE("entropy decode failed");
      if (cs + 1 > dst_capacity) return PIK_FAILURE("entropy decode failed");
      memcpy(dst, &src[*pos], ++cs);
      *pos += cs;
      *ds = cs;
    } else {
      if (*pos + cs > src_capacity) return PIK_FAILURE("entropy decode failed");
      if (!MaybeEntropyDecode(&src[*pos], cs, dst_capacity, dst, ds)) {
        return PIK_FAILURE("entropy decode failed");
      }
      *pos += cs;
    }

    return true;
  }

#define Update_Errors_0_1_2_3                                  \
  err = prediction0 - truePixelValue;                          \
  if (err < 0) err = -err; /* abs() and min()? worse speed! */ \
  if (err > kMaxError) err = kMaxError;                        \
  errors0[yp + x] = err;                                       \
  err = prediction1 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > kMaxError) err = kMaxError;                        \
  errors1[yp + x] = err;                                       \
  err = prediction2 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > kMaxError) err = kMaxError;                        \
  errors2[yp + x] = err;                                       \
  err = prediction3 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > kMaxError) err = kMaxError;                        \
  errors3[yp + x] = err;

#define Update_Size_And_Errors                                    \
  ++esize[maxErr];                                                \
  trueErr[yp + x] = err;                                          \
  err = numBits(err >= 0 ? err : -err);                           \
  nbitErr[yp + x] = (err <= WithSIGN ? err * 2 : err + WithSIGN); \
  Update_Errors_0_1_2_3

  const uint16_t smt0[64] = {
    0x2415, 0x1d7d, 0x1f71, 0x46fe, 0x24f1, 0x3f15, 0x4a65, 0x6236,
    0x242c, 0x34ce, 0x4872, 0x5cf6, 0x4857, 0x64fe, 0x6745, 0x7986,
    0x24ad, 0x343c, 0x499a, 0x5fb5, 0x49a9, 0x61e8, 0x6e1f, 0x78ae,
    0x4ba3, 0x6332, 0x6c8b, 0x7ccd, 0x6819, 0x8247, 0x83f2, 0x8cce,
    0x247e, 0x3277, 0x391f, 0x5ea3, 0x4694, 0x5168, 0x67e3, 0x784b,
    0x474b, 0x5072, 0x666b, 0x6cb3, 0x6514, 0x7ba6, 0x83e4, 0x8cef,
    0x48bf, 0x6363, 0x6677, 0x7b76, 0x67f9, 0x7e0d, 0x826f, 0x8a52,
    0x659f, 0x7d6f, 0x7f8e, 0x8f66, 0x7ed6, 0x9169, 0x9269, 0x90e4,
  };

  uint8_t* Palette_compress(int numChannels, uint32_t *numColors, uint8_t *pb,
                     std::vector<uint32_t> &palette123, uint32_t *firstColors) {
    for (int channel = 0; channel < numChannels; ++channel) {
      uint32_t *palette = &palette123[0x10000 * channel];
      uint8_t *pb0 = pb;
      pb += 2;  // reserve 2 bytes for  (Compressed Size)*2 + Method
      palette[firstColors[channel]] = 1;
      uint32_t nc = numColors[channel], x1 = 0, x2 = 0xffffffff;
      int x, smt[64], context6 = 0, sumv = 0;
      for (int i = 0; i < 64; ++i)  smt[i] = smt0[i] << 11;  // 1<<(15+11);
      for (x = 0; x < 0x10000; ++x) {
        int v = (palette[x] ? 1 : 0);
        uint32_t pr = smt[context6] >> 11;
        uint32_t xmid = x1 + ((x2-x1) >> 16)*pr + (((x2-x1) & 0xffff)*pr >> 16);
        assert(pr>=0 && pr<=0xffff && xmid>=x1 && xmid<x2);
        if (v) x2 = xmid;
        else   x1 = xmid + 1;
        if (((x1 ^ x2) & 0xff000000)==0) {
          do {
            *pb++ = x1 >> 24;
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
          }
          while (((x1 ^ x2) & 0xff000000)==0);
          if (pb >= pb0 + 2 + 0x10000)  break;
        }
        int p0 = smt[context6];
        p0 += ((v << (16+11)) - p0) * 5 >> 7;  // Learning rate
        smt[context6] = p0;
        context6 = (context6 * 2 + v) & 0x3f;
        sumv += v;
        if (sumv == nc || sumv + 0x10000 - 1 - x == nc)  break;
      }
      *pb++ = static_cast<uint8_t>((x1 >> 24) & 0xFF);
      //if (count > 512) {  for (int i = 0; i < 64; ++i)
      //                        printf("0x%x,", smt[i]);   printf("\n"); }
      int method = 0;
      if (pb - (pb0+2) >= ((x+7)>>3)) {   // Store, no compression
        method = 1;
        pb = pb0+2;
        for (int sumv = 0, x = 0; x < 0x10000; x += 8) {
          uint32_t b = 0, v;
          for (int y = x + 7; y >= x; --y)
              v = (palette[y] ? 1 : 0), b += b + v, sumv += v;
          *pb++ = b;
          if (sumv >= nc || sumv + 0x10000 - 8 - x == nc)  break;
        }
      }
      int compressedSize = (pb - (pb0+2)) * 2  +  method;
      pb0[0] = static_cast<uint8_t>( compressedSize       & 0xFF);
      pb0[1] = static_cast<uint8_t>((compressedSize >> 8) & 0xFF);
      palette[firstColors[channel]] = 0;
    }
    return pb;
  }

#define FWr(buf, bufsize) {                          \
    if (run == 0) {                                  \
      size_t current = bytes->size();                \
      bytes->resize(bytes->size() + bufsize);        \
      memcpy(bytes->data() + current, buf, bufsize); \
    }}                                               \

#define FWrByte(b) {  \
    uint8_t byte = b; \
    FWr(&byte, 1);    \
  }

  void PerChannelPalette_compress_1(ImageU& img, PaddedBytes* bytes, int run) {
    const int numChannels = 1, channel = 0;
    size_t xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint32_t> palette123(0x10000 * numChannels);
    memset(palette123.data(), 0, 0x10000 * numChannels * sizeof(uint32_t));
    uint8_t bits[0x10010 / 8], flags = 0, compressedData[6];
    memset(bits, 0, sizeof(bits));
    uint32_t firstColors[3], numColors[3] = {0xffff, 0xffff, 0xffff};

      uint32_t i, count, *palette = &palette123[0x10000 * channel];
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          palette[rowImg[x]] = 1;
      }
      // count the number of pixel values present in the channel
      for (i = 0; i < 0x10000; ++i)
        if (palette[i]) break;
      for (firstColors[channel] = i, count = 0; i < 0x10000; ++i)
        if (palette[i]) palette[i] = count++;
      // printf("count=%5d, %f%%\n", count, count * 100. / 65536);
      if (count > 65536 / 16) {  // TODO: smarter decision making
        flags = 0;
      }
      else flags += 1 << channel;
      numColors[channel] = count;

    FWrByte(flags);  // As of Jan.2019, ImageMethod==flags, either 0 or 7
    if (flags) {
      uint8_t *pb = Palette_compress(numChannels, &numColors[0], &bits[0],
                                     palette123, &firstColors[0]);
      // Apply the channel's "palette"
      uint32_t *palette = &palette123[0x10000 * channel];
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          rowImg[x] = palette[rowImg[x]];
      }
      int pos = encodeVarInt(numColors[channel], &compressedData[0]);
      FWr(&compressedData[0], pos);
      FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
    }  // if (flags)
  }

  void PerChannelPalette_compress_3(Image3U& img, PaddedBytes* bytes, int run) {
    const int numChannels = 3;
    size_t xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint32_t> palette123(0x10000 * numChannels);
    memset(palette123.data(), 0, 0x10000 * numChannels * sizeof(uint32_t));
    uint8_t bits[3 * 0x10010 / 8], flags = 0, compressedData[3*6];
    memset(bits, 0, sizeof(bits));
    uint32_t firstColors[3], numColors[3] = {0xffff, 0xffff, 0xffff};

    for (int channel = 0; channel < numChannels; ++channel) {
      uint32_t i, count, *palette = &palette123[0x10000 * channel];
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          palette[rowImg[x]] = 1;
      }
      // count the number of pixel values present in the channel
      for (i = 0; i < 0x10000; ++i)
        if (palette[i]) break;
      for (firstColors[channel] = i, count = 0; i < 0x10000; ++i)
        if (palette[i]) palette[i] = count++;
      // printf("count=%5d, %f%%\n", count, count * 100. / 65536);
      if (count > 65536 * 3 / 4) {  // TODO: smarter decision making
        flags = 0;
        break;
      }
      flags += 1 << channel;
      numColors[channel] = count;
    }  // for channel

    FWrByte(flags);  // As of Jan.2019, ImageMethod==flags, either 0 or 7
    if (flags) {
      uint8_t *pb = Palette_compress(numChannels, &numColors[0], &bits[0],
                                     palette123, &firstColors[0]);
      // Apply the channel's "palette"
      for (int channel = 0; channel < numChannels; ++channel) {
        uint32_t *palette = &palette123[0x10000 * channel];
        for (size_t y = 0; y < ysize; ++y) {
          uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
            rowImg[x] = palette[rowImg[x]];
        }
      }
      int pos = 0;
      for (int c=0; c < numChannels; ++c)
        pos += encodeVarInt(numColors[c], &compressedData[pos]);
      FWr(&compressedData[0], pos);
      FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
    }  // if (flags)
  }


  bool PerChannelPalette_decompress(const uint8_t* compressedData,
        size_t compressedSize, size_t *pos, int numChannels, int imageMethod,
        std::vector<int> &palette) {
    int numColors[3];
    for (int channel = 0; channel < numChannels; ++channel) {
      numColors[channel] = decodeVarInt(compressedData, compressedSize, pos);
      if (numColors[channel] > 65536)  return PIK_FAILURE("lossless16");
    }
    const uint8_t* p = &compressedData[*pos];
    const uint8_t* p_end = compressedData + compressedSize;
    for (int channel = 0; channel < numChannels; ++channel)
      if (imageMethod & (1 << channel)) {
        int methodAndSize = p[0] + p[1]*256, cSize = methodAndSize >> 1;
        p += 2;
        const uint8_t *p00 = p, *pcEnd = p00 + cSize;
        if (pcEnd >= p_end)  return PIK_FAILURE("lossless16");
        if (methodAndSize & 1) {
          int x=0, sumv = channel << 16, stop = sumv + numColors[channel];
          while (x < 0x10000) {
            if (p >= pcEnd)  return PIK_FAILURE("lossless16");
            for (int b = *p++, i = 0; i < 8; ++i)
              palette[sumv] = x++, sumv += b & 1, b >>= 1;
            if (sumv >= stop)  break;
            if (sumv + 0x10000 - x == stop) {
              while (x < 0x10000)  palette[sumv++] = x++;
            }
          } // while x
          continue;
        } // if (methodAndSize & 1)

        uint32_t smt[64], x1 = 0, x2 = 0xffffffff, xr = 0, context6 = 0,
                 sumv = channel << 16, stop = sumv + numColors[channel];
        for (int i = 0; i < 4; ++i)
          xr = (xr << 8) + (p >= pcEnd? 0xFF : *p++);
        for (int i = 0; i < 64; ++i)  smt[i] = smt0[i] << 11;
        for (int x = 0; x < 0x10000; ) {
          int v;
          uint32_t pr = smt[context6] >> 11;

          uint32_t xmid = x1 + ((x2-x1)>>16)*pr + (((x2-x1) & 0xffff)*pr >> 16);
          assert(pr>=0 && pr<=0xffff && xmid>=x1 && xmid<x2);
          if (xr <= xmid)  x2 = xmid, v = 1;   else  x1 = xmid + 1, v = 0;

          while (((x1 ^ x2) & 0xff000000)==0) {  // Binary arithm decomprs
            xr = (xr<<8) + (p >= pcEnd? 0xFF : *p++);
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
          }

          int p0 = smt[context6];
          p0 += ((v << (16+11)) - p0) * 5 >> 7;  // Learning rate
          smt[context6] = p0;
          context6 = (context6 * 2 + v) & 0x3f;
          palette[sumv] = x++;
          sumv += v;
          if (sumv == stop) break;
          if (sumv + 0x10000 - x == stop) {
              while (x < 0x10000)  palette[sumv++] = x++;
          }
        }  // for x
        p = p00 + cSize;
      } // if (imageMethod & ...
    *pos = p - &compressedData[0];
    return true;
  }






  bool Grayscale16bit_compress(const ImageU& img_in, PaddedBytes* bytes) {
    WithSIGN = kWithSign1, BitsMAX = kBitsMax1, NumCONTEXTS = kNumContexts1;

    // The code modifies the image for palette so must copy for now.
    ImageU img = CopyImage(img_in);

    size_t esize[kNumContexts1], xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint8_t> temp_buffer(kGroupSize2plus * 2);
    compressedData = temp_buffer.data();

    clock_t start = clock();
    for (int run = 0; run < kNumRuns; ++run) {
      size_t pos = encodeVarInt(xsize, &compressedData[0]);
      pos       += encodeVarInt(ysize, &compressedData[pos]);
      FWr(&compressedData[0], pos);

      if (xsize * ysize > 256 * 256)  // TODO: smarter decision making here
        PerChannelPalette_compress_1(img, bytes, run);

      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          memset(esize, 0, sizeof(esize));
          for (size_t y = 0, yp = 0, yp1,
                      yEnd = std::min((size_t)kGroupSize, ysize - groupY);
               y < yEnd; ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
            rowImg = img.Row(groupY + y) + groupX;
            rowPrev = (y == 0 ? nullptr : img.Row(groupY + y - 1) + groupX);
            width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
            for (size_t x = 0; x <= width; ++x) {
              int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
              assert(0 <= maxErr && maxErr <= kNumContexts1 - 1);
              assert(0 <= prediction && prediction <= 0xffff);

              int truePixelValue = (int)rowImg[x];
              int err = prediction - truePixelValue;
              size_t s = esize[maxErr];
              edata[maxErr][s] = sign_LSB_forward_transform[err & 0xffff];

              Update_Size_And_Errors
            }  // x
          }    // y
          size_t pos = 0;
          for (int i = 0; i < kNumContexts1; ++i) {
            size_t S = esize[i];
            if (S == 0) {
              // This means uncompressed size 0.
              pos += encodeVarInt(0, &compressedData[pos]);
              continue;
            }
            uint16_t* d = &edata[i][0];
            // first, compress MSBs (most significant bytes)
            uint8_t* p = &compressedData[pos + 8];
            for (size_t x = 0; x < S; ++x) p[x] = d[x] >> 8;
            PIK_RETURN_IF_ERROR(
              compressWithEntropyCode(&pos, S, compressedData));

            if (i > 9 || S < 128) {  //  9  128
              // then, compress LSBs (least significant bytes)
              p = &compressedData[pos + 8];
              for (size_t x = 0; x < S; ++x) p[x] = d[x] & 255;  // All
              PIK_RETURN_IF_ERROR(
                compressWithEntropyCode(&pos, S, compressedData));
            } else {
              p = &compressedData[pos + 8];
              size_t y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] < 256) p[y++] = d[x] & 255;  // LSBs such that MSB==0
              if (y) {
                PIK_RETURN_IF_ERROR(
                  compressWithEntropyCode(&pos, y, compressedData));
              }

              p = &compressedData[pos + 8];
              y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] >= 256) p[y++] = d[x] & 255;  // LSBs such that MSB!=0
              if (y) {
                PIK_RETURN_IF_ERROR(
                  compressWithEntropyCode(&pos, y, compressedData));
              }
            }  // if (i > 9)
          }    // i
          FWr(&compressedData[0], pos)
        }  // groupX
      }    // groupY
    }      // run
    if (kNumRuns > 1)
      printf("%d runs, %1.5f seconds", kNumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    return true;
  }

  bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                 ImageU* result) {
    WithSIGN = kWithSign1, BitsMAX = kBitsMax1, NumCONTEXTS = kNumContexts1;
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    // Size of an edata entry
    size_t maxDecodedSize = kGroupSize * kGroupSize;
    // Size of a compressedDataTmpBuf entry
    size_t maxDecodedSize2 = kGroupSize2plus;

    size_t esize[kNumContexts1], xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = decodeVarInt(compressedData, compressedSize, &pos0);
    ysize = decodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 134217728ull || ysize >= 65536) {
      return PIK_FAILURE("lossless16");
    }
    pik::ImageU img(xsize, ysize);
    std::vector<int> palette(0x10000);

    clock_t start = clock();
    for (int run = 0; run < kNumRuns; ++run) {
      size_t pos = pos0;
      if (xsize * ysize > 256 * 256) {  // TODO: smarter decision making here
        imageMethod = compressedData[pos++];
        if (imageMethod) {  // As of Jan.2019, ImageMethod is either 0 or 7
          PIK_RETURN_IF_ERROR(
            PerChannelPalette_decompress(compressedData, compressedSize,
                                            &pos, 1, imageMethod, palette));
        }
      }  // if (xsize*ysize ...
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          size_t decompressedSize = 0;  // is used only for return PIK_FAILURE

          for (int i = 0; i < kNumContexts1; ++i) {
            size_t ds, ds1, ds2, ds3;
            // first, decompress MSBs (most significant bytes)
            PIK_RETURN_IF_ERROR(
              decompressWithEntropyCode((uint8_t*)&edata[i][0],
                                           maxDecodedSize, compressedData,
                                           compressedSize, &ds1, &pos));
            if (!ds1) continue;
            if (i > 9 || ds1 < 128) {  // All LSBs at once
              PIK_RETURN_IF_ERROR(
                decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                             maxDecodedSize2, compressedData,
                                             compressedSize, &ds2, &pos));
              if (ds1 != ds2) return PIK_FAILURE("lossless16");
              uint16_t* dst = &edata[i][0];
              uint8_t* p = (uint8_t*)dst;
              for (int j = ds1 - 1; j >= 0; --j)
                dst[j] = p[j] * 256 + compressedDataTmpBuf[j];  // MSB*256 + LSB
            } else {
              uint16_t* dst = &edata[i][0];
              uint8_t* p = (uint8_t*)dst;
              ds2 = ds3 = 0;
              for (int j = ds1 - 1; j >= 0; --j)
                if (p[j])
                  ++ds3;
                else
                  ++ds2;

              if (ds2) {  // LSBs such that MSB==0
                PIK_RETURN_IF_ERROR(
                  decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                               maxDecodedSize2, compressedData,
                                               compressedSize, &ds, &pos));
                if (ds != ds2) return PIK_FAILURE("lossless16");
              }

              if (ds3) {  // LSBs such that MSB!=0
                PIK_RETURN_IF_ERROR(
                  decompressWithEntropyCode(&compressedDataTmpBuf[ds2],
                                               maxDecodedSize2, compressedData,
                                               compressedSize, &ds, &pos));
                if (ds != ds3) return PIK_FAILURE("lossless16");
              }
              uint8_t *p2 = &compressedDataTmpBuf[ds2 - 1],
                      *p3 = &compressedDataTmpBuf[ds1 - 1];  // Note ds1=ds2+ds3
              for (int j = ds1 - 1; j >= 0; --j)
                dst[j] = p[j] * 256 + (p[j] == 0 ? *p2-- : *p3--);
            }
            decompressedSize += ds1;
          }  // for i
          if (!(decompressedSize ==
                std::min((size_t)kGroupSize, ysize - groupY) *
                std::min((size_t)kGroupSize, xsize - groupX))) {
            return PIK_FAILURE("lossless16");
          }
// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.

          memset(esize, 0, sizeof(esize));
          for (size_t y = 0,
                      yEnd = std::min((size_t)kGroupSize, ysize - groupY),
                      yp = 0, yp1;
               y < yEnd; ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
            rowImg = img.Row(groupY + y) + groupX;
            rowPrev = (y == 0 ? nullptr : img.Row(groupY + y - 1) + groupX);
            width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
            for (size_t x = 0; x <= width; ++x) {
              int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
              assert(0 <= maxErr && maxErr <= kNumContexts1 - 1);
              assert(0 <= prediction && prediction <= 0xffff);

              size_t s = esize[maxErr];
              int err = edata[maxErr][s];
              int truePixelValue =
                  (prediction - sign_LSB_backward_transform[err]) & 0xffff;
              rowImg[x] = truePixelValue;
              err = prediction - truePixelValue;

              Update_Size_And_Errors
            }  // x
          }    // y
        }      // groupX
      }        // groupY
      *bytes_pos += pos;
      if (imageMethod & 1) {
        for (size_t y = 0; y < ysize; ++y) {
          uint16_t* const PIK_RESTRICT rowImg = img.Row(y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
            rowImg[x] = palette[rowImg[x]];
        }
      }
    }  // run
    if (kNumRuns > 1)
      printf("%d runs, %1.5f seconds", kNumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    *result = std::move(img);
    return true;
  }




  const int PL1 = 0, PL2 = 1, PL3 = 2;

  enum PlaneMethods_30 {  // 8/30 are redundant (left for encoder's convenience)
    RR_G_B = 0,           // p1=R  p2=G  p3=B
    RR_GmR_B = 1,         // p2-p1  p3
    RR_G_BmR = 2,         //   p2  p3-p1
    RR_GmR_BmR = 3,       // p2-p1 p3-p1

    RR_GmB_B = 4,  // == 21   p2-p3 @ p2
    RR_G_GmB = 5,  // ~= 12   p2-p3 @ p3

    RR_GmR_Bm2 = 6,  //  p2-p1  p3-(p1+p2)/2
    RR_Gm2_BmR = 7,  // p2-(p1+p3)/2   p3-p1
    RR_G_Bm2 = 8,    //   p2    p3-(p1+p2)/2
    RR_Gm2_B = 9,    // p2-(p1+p3)/2     p3

    R_GG_B = 10,  // p1=G  p2=R  p3=B
    RmG_GG_B = 11,
    R_GG_BmG = 12,
    RmG_GG_BmG = 13,

    RmB_GG_B = 14,  // == 22
    R_GG_RmB = 15,  // ~=  2

    RmG_GG_Bm2 = 16,
    Rm2_GG_BmG = 17,
    R_GG_Bm2 = 18,
    Rm2_GG_B = 19,

    R_G_BB = 20,  // p1=B  p2=R  p3=G
    R_GmB_BB = 21,
    RmB_G_BB = 22,
    RmB_GmB_BB = 23,

    RmG_G_BB = 24,  // == 11
    R_RmG_BB = 25,  // ~=  1

    RmB_Gm2_BB = 26,
    Rm2_GmB_BB = 27,
    R_Gm2_BB = 28,
    Rm2_G_BB = 29,
  };

  bool dcmprs512x512(pik::Image3U* img, int planeToDecompress, size_t& pos,
                     size_t groupY, size_t groupX,
                     const uint8_t* compressedData, size_t compressedSize) {
    // Size of an edata entry
    const size_t maxDecodedSize = kGroupSize * kGroupSize;
    // Size of a compressedDataTmpBuf entry
    const size_t maxDecodedSize2 = kGroupSize2plus;

    size_t esize[kNumContexts3], xsize = img->xsize(), ysize = img->ysize();
    size_t decompressedSize = 0;  // is used only for 'return PIK_FAILURE'
    memset(esize, 0, sizeof(esize));
    for (int i = 0; i < kNumContexts3; ++i) {
      size_t ds, ds1, ds2, ds3;
      // first, decompress MSBs (most significant bytes)
      PIK_RETURN_IF_ERROR(
        decompressWithEntropyCode((uint8_t*)&edata[i][0], maxDecodedSize,
                               compressedData, compressedSize, &ds1, &pos));
      if (!ds1) continue;
      uint32_t freq[256];
      memset(freq, 0, sizeof(freq));
      uint16_t* dst = &edata[i][0];
      uint8_t* p = (uint8_t*)dst;
      for (int j = 0; j < ds1; ++j) ++freq[p[j]];

      if (ds1 < 120 || freq[0] < 120) {  // All LSBs at once
        PIK_RETURN_IF_ERROR(
          decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                       maxDecodedSize2, compressedData,
                                       compressedSize, &ds2, &pos));
        if (ds1 != ds2) return PIK_FAILURE("lossless16");
        for (int j = ds1 - 1; j >= 0; --j)
          dst[j] = p[j] * 256 + compressedDataTmpBuf[j];  // MSB*256 + LSB
      } else {
        uint32_t c = (freq[0] > (ds1 * 13 >> 4) ? 2 : 1);
        ds2 = freq[0] + (c == 2 ? freq[1] : 0);
        ds3 = ds1 - ds2;
        if (ds2) {  // LSBs such that MSB==0
          PIK_RETURN_IF_ERROR(
            decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                         maxDecodedSize2, compressedData,
                                         compressedSize, &ds, &pos));
          if (ds != ds2) return PIK_FAILURE("lossless16");
        }

        if (ds3) {  // LSBs such that MSB!=0
          PIK_RETURN_IF_ERROR(
            decompressWithEntropyCode(&compressedDataTmpBuf[ds2],
                                         maxDecodedSize2, compressedData,
                                         compressedSize, &ds, &pos));
          if (ds != ds3) return PIK_FAILURE("lossless16");
        }
        uint8_t *p2 = &compressedDataTmpBuf[ds2 - 1],
                *p3 = &compressedDataTmpBuf[ds1 - 1];  // Note ds1=ds2+ds3
        for (int j = ds1 - 1; j >= 0; --j)
          dst[j] = p[j] * 256 + (p[j] < c ? *p2-- : *p3--);
      }
      decompressedSize += ds1;
    }  // for i
    if (decompressedSize !=
          std::min((size_t)kGroupSize, ysize - groupY) *
          std::min((size_t)kGroupSize, xsize - groupX)) {
      return PIK_FAILURE("lossless16");
    }

    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift = (area > 25600 ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;

    for (size_t y = 0, yp = 0, yp1; y < yEnd;
         ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
      rowImg = img->PlaneRow(planeToDecompress, groupY + y) + groupX;
      rowPrev =
          (y == 0 ? nullptr
                  : img->PlaneRow(planeToDecompress, groupY + y - 1) + groupX);
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        assert(0 <= maxErr && maxErr <= kNumContexts3 - 1);
        assert(0 <= prediction && prediction <= 0xffff);

        size_t s = esize[maxErr];
        int err = edata[maxErr][s], truePixelValue =
            (prediction - sign_LSB_backward_transform[err]) & 0xffff;
        rowImg[x] = truePixelValue;
        err = prediction - truePixelValue;

        Update_Size_And_Errors
      }  // x
    }    // y
    return true;
  }

  bool Colorful16bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                Image3U* result) {
    WithSIGN = kWithSign3, BitsMAX = kBitsMax3, NumCONTEXTS = kNumContexts3;
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t cSize = bytes.size() - *bytes_pos;
    const uint8_t* cprsdData = bytes.data() + *bytes_pos;

    size_t xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = decodeVarInt(cprsdData, cSize, &pos0);
    ysize = decodeVarInt(cprsdData, cSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 134217728ull || ysize >= 65536)
      return PIK_FAILURE("lossless16");

    pik::Image3U img(xsize, ysize);
    std::vector<int> palette(0x10000 * 3);

    clock_t start = clock();
    for (int run = 0; run < kNumRuns; ++run) {
      size_t pos = pos0;
      if (pos >= cSize)
        return PIK_FAILURE("lossless16: out of bounds");
      if (xsize * ysize > 256 * 256) {  // TODO: smarter decision making here
        imageMethod = cprsdData[pos++];
        if (imageMethod) {  // As of Jan.2019, ImageMethod is either 0 or 7
          PIK_RETURN_IF_ERROR(
            PerChannelPalette_decompress(cprsdData, cSize, &pos, 3,
               imageMethod, palette));
        }
      }  // if (xsize*ysize ...

      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          uint16_t *PIK_RESTRICT row1, *PIK_RESTRICT row2, *PIK_RESTRICT row3;
          size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
          size_t xEnd = std::min((size_t)kGroupSize, xsize - groupX);
          PIK_RETURN_IF_ERROR(
            dcmprs512x512(&img, PL1, pos, groupY, groupX, cprsdData, cSize));
          PIK_RETURN_IF_ERROR(
            dcmprs512x512(&img, PL2, pos, groupY, groupX, cprsdData, cSize));
          PIK_RETURN_IF_ERROR(
            dcmprs512x512(&img, PL3, pos, groupY, groupX, cprsdData, cSize));

#define T3bgn                                      \
  for (size_t y = 0; y < yEnd; ++y) {              \
    row1 = img.PlaneRow(PL1, groupY + y) + groupX; \
    row2 = img.PlaneRow(PL2, groupY + y) + groupX; \
    row3 = img.PlaneRow(PL3, groupY + y) + groupX; \
    for (size_t x = 0; x < xEnd; ++x) {            \
      int R = row1[x], G = row2[x], B = row3[x];   \
      (void)R;                                     \
      (void)G;                                     \
      (void)B;

// Close T3bgn above; not using a #define confuses brace matching of editor.
#define CC \
  }        \
  }

          if (pos >= cSize)
            return PIK_FAILURE("lossless16: out of bounds");
          int planeMethod = cprsdData[pos++];
          switch (planeMethod) {
            case 0:
            case 10:
            case 20:
              break;
            case 1:
              T3bgn G += R + 0x8000;
              row2[x] = G;
              CC break;
            case 2:
              T3bgn B += R + 0x8000;
              row3[x] = B;
              CC break;
            case 3:
              T3bgn G += R + 0x8000;
              B += R + 0x8000;
              row2[x] = G;
              row3[x] = B;
              CC break;
            case 22:
            case 4:
              T3bgn row2[x] = G + B + 0x8000;
              CC break;
            case 5:
              T3bgn row3[x] = G - B + 0x8000;
              CC break;
            case 6:
              T3bgn row2[x] = G = (G + R + 0x8000) & 0xffff;
              row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 7:
              T3bgn row3[x] = B = (B + R + 0x8000) & 0xffff;
              row2[x] = G + ((R + B) >> 1) + 0x8000;
              CC break;
            case 8:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 9:
              T3bgn row2[x] = G + ((R + B) >> 1) + 0x8000;
              CC break;

            case 24:
            case 11:
              T3bgn R += G + 0x8000;
              row1[x] = R;
              CC break;
            case 12:
              T3bgn B += G + 0x8000;
              row3[x] = B;
              CC break;
            case 13:
              T3bgn R += G + 0x8000;
              B += G + 0x8000;
              row1[x] = R;
              row3[x] = B;
              CC break;
            case 21:
            case 14:
              T3bgn row1[x] = R + B + 0x8000;
              CC break;
            case 15:
              T3bgn row3[x] = R - B + 0x8000;
              CC break;

            case 16:
              T3bgn row1[x] = R = (R + G + 0x8000) & 0xffff;
              row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 17:
              T3bgn row3[x] = B = (B + G + 0x8000) & 0xffff;
              row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;
            case 18:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 19:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;

            case 23:
              T3bgn G += B + 0x8000;
              R += B + 0x8000;
              row1[x] = R;
              row2[x] = G;
              CC break;
            case 25:
              T3bgn row2[x] = R - G + 0x8000;
              CC break;
            case 26:
              T3bgn row1[x] = R = (R + B + 0x8000) & 0xffff;
              row2[x] = G + ((B + R) >> 1) + 0x8000;
              CC break;
            case 27:
              T3bgn row2[x] = G = (G + B + 0x8000) & 0xffff;
              row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;
            case 28:
              T3bgn row2[x] = G + ((B + R) >> 1) + 0x8000;
              CC break;
            case 29:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;
          }
        }  // groupX
      }    // groupY
// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.

      for (int channel = 0; channel < 3; ++channel)
        if (imageMethod & (1 << channel)) {
          int* p = &palette[0x10000 * channel];
          for (size_t y = 0; y < ysize; ++y) {
            uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
            for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
              rowImg[x] = p[rowImg[x]];
          }
        }
      *bytes_pos += pos;
    }  // run
    if (kNumRuns > 1)
      printf("%d runs, %1.5f seconds", kNumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    *result = std::move(img);
    return true;
  }

  bool cmprs512x512(pik::Image3U& img, int planeToCompress, int planeToUse,
                        size_t groupY, size_t groupX,
                        uint8_t* compressedOutput, size_t *csize) {
    size_t esize[kNumContexts3], xsize = img.xsize(), ysize = img.ysize();
    memset(esize, 0, sizeof(esize));
    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift = (area > 25600 ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;

    for (size_t y = 0, yp = 0, yp1; y < yEnd;
         ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
      rowImg = img.PlaneRow(planeToCompress, groupY + y) + groupX;
      rowPrev = (!y ? nullptr :
               img.PlaneRow(planeToCompress, groupY + y - 1) + groupX);
      uint16_t* PIK_RESTRICT rowUse =
               img.PlaneRow(planeToUse, groupY + y) + groupX;
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        assert(0 <= maxErr && maxErr <= kNumContexts3 - 1);
        assert(0 <= prediction && prediction <= 0xffff);
        int truePixelValue = (int)rowImg[x];
        if (planeToCompress != planeToUse) {
          truePixelValue -= (int)rowUse[x] - 0x8000;
          truePixelValue &= 0xffff;
          rowImg[x] = truePixelValue;
        }
        int err = prediction - truePixelValue;
        size_t s = esize[maxErr];
        edata[maxErr][s] = sign_LSB_forward_transform[err & 0xffff];
        Update_Size_And_Errors
      }  // x
    }    // y

    size_t pos = 0;
    for (int i = 0; i < kNumContexts3; ++i) {
      size_t c = 0, S = esize[i];
      if (S == 0) {  // If uncompressed size is 0: empty bucket.
        pos += encodeVarInt(0, &compressedOutput[pos]);
        continue;
      }
      uint16_t* d = &edata[i][0];
      // first, compress MSBs (most significant bytes)
      uint8_t* p = &compressedOutput[pos + 8];
      for (size_t x = 0; x < S; ++x) p[x] = d[x] >> 8, c += (p[x] ? 0 : 1);
      PIK_RETURN_IF_ERROR(
        compressWithEntropyCode(&pos, S, compressedOutput));
      if (S < 120 || c < 120) {  // 120
        // then, compress LSBs (least significant bytes)
        p = &compressedOutput[pos + 8];
        for (size_t x = 0; x < S; ++x) p[x] = d[x] & 255;  // All LSBs!
        PIK_RETURN_IF_ERROR(
          compressWithEntropyCode(&pos, S, compressedOutput));
      } else {
        c = (c > (S * 13 >> 4) ? 2 : 1) << 8;
        p = &compressedOutput[pos + 8];
        size_t y = 0;
        for (size_t x = 0; x < S; ++x)
          if (d[x] < c) p[y++] = d[x] & 255;  // LSBs such that MSB<2
        if (y) {
          PIK_RETURN_IF_ERROR(
            compressWithEntropyCode(&pos, y, compressedOutput));
        }

        p = &compressedOutput[pos + 8];
        y = 0;
        for (size_t x = 0; x < S; ++x)
          if (d[x] >= c) p[y++] = d[x] & 255;  // LSBs such that MSB>=2
        if (y) {
          PIK_RETURN_IF_ERROR(
            compressWithEntropyCode(&pos, y, compressedOutput));
        }
      }  // if (S < 120)
    }    // for i
    *csize = pos;
    return true;
  }

  bool Colorful16bit_compress(const Image3U& img_in, PaddedBytes* bytes) {
    WithSIGN = kWithSign3, BitsMAX = kBitsMax3, NumCONTEXTS = kNumContexts3;
    clock_t start = clock();

    // The code modifies the image for palette so must copy for now.
    Image3U img = CopyImage(img_in);

    std::vector<uint8_t> temp_buffer(kGroupSize2plus * 2 * 6);
    compressedData = temp_buffer.data();

    for (int run = 0; run < kNumRuns; ++run) {
      size_t xsize = img.xsize(), ysize = img.ysize(), pos;
      pos =  encodeVarInt(xsize, &compressedData[0]);
      pos += encodeVarInt(ysize, &compressedData[pos]);
      FWr(&compressedData[0], pos);

      if (xsize * ysize > 256 * 256)  // TODO: smarter decision making here
        PerChannelPalette_compress_3(img, bytes, run);

      uint8_t* compressedData2 = &compressedData[kGroupSize2plus * 2];
      uint8_t* compressedData3 = &compressedData[kGroupSize2plus * 4];
      uint8_t* cd4 = &compressedData[kGroupSize2plus * 6];
      uint8_t* cd5 = &compressedData[kGroupSize2plus * 8];
      uint8_t* cd6 = &compressedData[kGroupSize2plus * 10];
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          size_t S1, S2, S3, S4, S5, S6, s1, s2, s3, p1, p2, p3;
          uint8_t *cd1, *cd2, *cd3;
          int planeMethod;  // Here we try guessing which of the 30 PlaneMethods
                            // is best, after trying just six color planes.
          PIK_RETURN_IF_ERROR(
            cmprs512x512(img, PL1, PL1, groupY, groupX, compressedData,  &s1));
          PIK_RETURN_IF_ERROR(
            cmprs512x512(img, PL2, PL2, groupY, groupX, compressedData2, &s2));
          PIK_RETURN_IF_ERROR(
            cmprs512x512(img, PL3, PL3, groupY, groupX, compressedData3, &s3));

          S1 = s2, p1 = PL2, cd1 = compressedData2, planeMethod = 10;
          S2 = s1, p2 = PL1, cd2 = compressedData;
          S3 = s3, p3 = PL3, cd3 = compressedData3;
          if (s1 < s2 * 63 / 64 && s1 < s3) {
            S1 = s1, p1 = PL1, cd1 = compressedData, planeMethod = 0;
            S2 = s2, p2 = PL2, cd2 = compressedData2;
            S3 = s3, p3 = PL3, cd3 = compressedData3;
          } else if (s3 < s2 * 63 / 64 && s3 < s1) {
            S1 = s3, p1 = PL3, cd1 = compressedData3, planeMethod = 20;
            S2 = s1, p2 = PL1, cd2 = compressedData;
            S3 = s2, p3 = PL2, cd3 = compressedData2;
          }
          PIK_RETURN_IF_ERROR(
            cmprs512x512(img, p2, p1, groupY, groupX, cd4, &S4)); // R-G+0x8000
          PIK_RETURN_IF_ERROR(
            cmprs512x512(img, p3, p1, groupY, groupX, cd5, &S5)); // B-G+0x8000
          if (p1 == PL1)
            FWr(cd1, S1)

          if (S4 >= S2 && S5 >= S3) {
            PIK_RETURN_IF_ERROR(
              cmprs512x512(img, p2, p3, groupY, groupX, cd6, &S6)); // R-B+0x..
            if (S6 >= S2 && S6 >= S3)     FWr(cd2, S2)
            else if (S3 > S2 && S3 > S6)  FWr(cd2, S2)
            else                          FWr(cd6, S6)
            if (p1 == PL2)  FWr(cd1, S1)
            if (S6 >= S2 && S6 >= S3)    { FWr(cd3, S3) }
            else if (S3 > S2 && S3 > S6) { FWr(cd6, S6) planeMethod+=5; }
            else                         { FWr(cd3, S3) planeMethod+=4; }
          }
          else {
            size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY) + groupY;
            size_t xEnd = std::min((size_t)kGroupSize, xsize - groupX);
            size_t pp = S5 < S4? p2 : p3;
            for (size_t y = groupY; y < yEnd; ++y) {
                uint16_t* PIK_RESTRICT row1 = img.PlaneRow(p1, y) + groupX;
                uint16_t* PIK_RESTRICT row2 = img.PlaneRow(pp, y) + groupX;
                for (size_t x = 0; x < xEnd; ++x) {
                  uint32_t v1 = row1[x], v2 = (row2[x] + v1 + 0x8000) & 0xffff;
                  row2[x] = ((v1 + v2) >> 1) - v1 + 0x8000;
                }
            }
            if (S5 < S4) {
              PIK_RETURN_IF_ERROR(
                cmprs512x512(img, p3, p2, groupY, groupX, cd6, &S6)); //B-RpG/2
              if (S4 < S2)  FWr(cd4, S4) else  FWr(cd2, S2)
              if (p1 == PL2)
                FWr(cd1, S1)
              if (S3 <= S5 && S3 <= S6) {
                FWr(cd3, S3) planeMethod += 1;
              } else if (S5 <= S6) {
                FWr(cd5, S5) planeMethod += (S4 < S2 ? 3 : 2);
              } else {
                FWr(cd6, S6) planeMethod += (S4 < S2 ? 6 : 8);
              }
            } else {
              PIK_RETURN_IF_ERROR(
                cmprs512x512(img, p2, p3, groupY, groupX, cd6, &S6)); //R-BpG/2
              if (S2 <= S4 && S2 <= S6) {
                FWr(cd2, S2) planeMethod += 2;
              } else if (S4 <= S6) {
                FWr(cd4, S4) planeMethod += (S5 < S3 ? 3 : 1);
              } else {
                FWr(cd6, S6) planeMethod += (S5 < S3 ? 7 : 9);
              }
              if (p1 == PL2)
                FWr(cd1, S1)
              if (S5 < S3) FWr(cd5, S5) else FWr(cd3, S3)
            }
          }
          if (p1 == PL3)
            FWr(cd1, S1)
          FWrByte(planeMethod);  // printf("%2d ", planeMethod);
        }                                       // groupX
      }                                         // groupY
    }                                           // run
    if (kNumRuns > 1)
      printf("%d runs, %1.5f seconds", kNumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    return true;
  }
};  // struct State

}  // namespace

bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale16bit_compress(img, bytes);
}

bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                               ImageU* result) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale16bit_decompress(bytes, pos, result);
}

bool Colorful16bit_compress(const Image3U& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Colorful16bit_compress(img, bytes);
}

bool Colorful16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              Image3U* result) {
  std::unique_ptr<State> state(new State());
  return state->Colorful16bit_decompress(bytes, pos, result);
}

}  // namespace pik
