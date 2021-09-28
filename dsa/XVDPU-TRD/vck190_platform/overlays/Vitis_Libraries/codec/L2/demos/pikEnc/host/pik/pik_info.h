// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PIK_INFO_H_
#define PIK_PIK_INFO_H_

// Optional output information for debugging and analyzing size usage.

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>
#include "pik/adaptive_reconstruction_fwd.h"
#include "pik/chroma_from_luma_fwd.h"
#include "pik/codec.h"
#include "pik/image.h"
#include "pik/pik_inspection.h"

namespace pik {

struct PikImageSizeInfo {
  PikImageSizeInfo() {}

  void Assimilate(const PikImageSizeInfo& victim) {
    num_clustered_histograms += victim.num_clustered_histograms;
    histogram_size += victim.histogram_size;
    entropy_coded_bits += victim.entropy_coded_bits;
    extra_bits += victim.extra_bits;
    total_size += victim.total_size;
    clustered_entropy += victim.clustered_entropy;
  }
  void Print(size_t num_inputs) const {
    printf("%10zd", total_size);
    if (histogram_size > 0) {
      printf("   [%6.2f %8zd %8zd %8zd %12.3f",
             num_clustered_histograms * 1.0 / num_inputs, histogram_size,
             entropy_coded_bits >> 3, extra_bits >> 3,
             histogram_size + (clustered_entropy + extra_bits) / 8.0f);
      printf("]");
    }
    printf("\n");
  }
  size_t num_clustered_histograms = 0;
  size_t histogram_size = 0;
  size_t entropy_coded_bits = 0;
  size_t extra_bits = 0;
  size_t total_size = 0;
  double clustered_entropy = 0.0f;
};

enum {
  kLayerHeader = 0,
  kLayerQuant,
  kLayerDequantTables,
  kLayerOrder,
  kLayerDC,
  kLayerCmap,
  kLayerControlFields,
  kLayerAC,
  kLayerDictionary,
  kNumImageLayers
};
static const char* kImageLayers[kNumImageLayers] = {
    "header", "quant",   "tables", "order",     "DC",
    "cmap",   "cfields", "AC",     "dictionary"};

struct TestingAux {
  Image3F* ac_prediction = nullptr;
};

// Metadata and statistics gathered during compression or decompression.
struct PikInfo {
  PikInfo() : layers(kNumImageLayers) {}

  PikInfo(const PikInfo&) = default;

  void Assimilate(const PikInfo& victim) {
    for (int i = 0; i < layers.size(); ++i) {
      layers[i].Assimilate(victim.layers[i]);
    }
    num_blocks += victim.num_blocks;
    num_dct2_blocks += victim.num_dct2_blocks;
    num_dct4_blocks += victim.num_dct4_blocks;
    num_dct16_blocks += victim.num_dct16_blocks;
    num_dct32_blocks += victim.num_dct32_blocks;
    entropy_estimate += victim.entropy_estimate;
    num_butteraugli_iters += victim.num_butteraugli_iters;
    cfl_stats_dc.Assimilate(victim.cfl_stats_dc);
    cfl_stats_ac.Assimilate(victim.cfl_stats_ac);
    adaptive_reconstruction_aux.Assimilate(victim.adaptive_reconstruction_aux);
  }

  PikImageSizeInfo TotalImageSize() const {
    PikImageSizeInfo total;
    for (int i = 0; i < layers.size(); ++i) {
      total.Assimilate(layers[i]);
    }
    return total;
  }

  void Print(size_t num_inputs) const {
    if (num_inputs == 0) return;
    printf("Average butteraugli iters: %10.2f\n",
           num_butteraugli_iters * 1.0 / num_inputs);
    for (int i = 0; i < layers.size(); ++i) {
      if (layers[i].total_size > 0) {
        printf("Total layer size %-10s\t", kImageLayers[i]);
        printf("%10f%%",
               100.0f * layers[i].total_size / TotalImageSize().total_size);
        layers[i].Print(num_inputs);
      }
    }
    printf("Total image size           ");
    TotalImageSize().Print(num_inputs);

    printf("\nCFL:\n");
    cfl_stats_dc.Print();
    cfl_stats_ac.Print();

    printf("\nAR:\n");
    adaptive_reconstruction_aux.Print();
  }

  template <typename T>
  void DumpImage(const char* label, const Image3<T>& image) const {
    if (debug_prefix.empty()) return;
    std::ostringstream pathname;
    pathname << debug_prefix << label << ".png";
    CodecContext context;
    CodecInOut io(&context);
    io.SetFromImage(StaticCastImage3<float>(image), context.c_srgb[0]);
    (void)io.EncodeToFile(io.c_current(), sizeof(T) * kBitsPerByte,
                          pathname.str());
  }
  template <typename T>
  void DumpImage(const char* label, const Image<T>& image) {
    DumpImage(label,
              Image3<T>(CopyImage(image), CopyImage(image), CopyImage(image)));
  }

  // This dumps coefficients as a 16-bit PNG with coefficients of a block placed
  // in the area that would contain that block in a normal image. To view the
  // resulting image manually, rescale intensities by using:
  // $ convert -auto-level IMAGE.PNG - | display -
  void DumpCoeffImage(const char* label, const Image3S& coeff_image) const;

  void SetInspectorImage3F(pik::InspectorImage3F inspector) {
    inspector_image3f_ = inspector;
  }

  // Allows hooking intermediate data inspection into various
  // places of the PIK processing pipeline. Returns true iff
  // processing should proceed.
  bool InspectImage3F(const char* label, const Image3F& image) {
    if (inspector_image3f_ != nullptr) {
      return inspector_image3f_(label, image);
    }
    return true;
  }

  std::vector<PikImageSizeInfo> layers;
  size_t num_blocks = 0;
  // Number of blocks that use larger DCT. Only set in the encoder.
  size_t num_dct2_blocks = 0;
  size_t num_dct4_blocks = 0;
  size_t num_dct16_blocks = 0;
  size_t num_dct32_blocks = 0;
  // Estimate of compressed size according to entropy-given lower bounds.
  float entropy_estimate = 0;
  int num_butteraugli_iters = 0;
  // If not empty, additional debugging information (e.g. debug images) is
  // saved in files with this prefix.
  std::string debug_prefix;

  // By how much the decoded image was downsampled relative to the encoded
  // image.
  size_t downsampling = 1;

  AdaptiveReconstructionAux adaptive_reconstruction_aux;
  CFL_Stats cfl_stats_dc;
  CFL_Stats cfl_stats_ac;

  pik::InspectorImage3F inspector_image3f_;

  // WARNING: this is actually an INPUT to some code, and must be
  // copy-initialized from aux_out to aux_outs.
  TestingAux testing_aux;
};

// Used to skip image creation if they won't be written to debug directory.
static inline bool WantDebugOutput(const PikInfo* info) {
  // Need valid pointer and filename.
  return info != nullptr && !info->debug_prefix.empty();
}

}  // namespace pik

#endif  // PIK_PIK_INFO_H_
