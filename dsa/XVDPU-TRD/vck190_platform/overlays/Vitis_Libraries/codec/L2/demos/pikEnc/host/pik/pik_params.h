// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PIK_PARAMS_H_
#define PIK_PIK_PARAMS_H_

// Parameters and flags that govern PIK compression/decompression.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace pik {

// Reasonable default for sRGB, matches common monitors. Butteraugli was tuned
// for this, we scale darker/brighter inputs accordingly.
static constexpr int kDefaultIntensityTarget = 250;
static constexpr float kIntensityMultiplier = 1.0f / kDefaultIntensityTarget;

// No effect if kDefault, otherwise forces a feature (typically a GroupHeader
// flag) on or off.
enum class Override : int { kOn = 1, kOff = 0, kDefault = -1 };

static inline Override OverrideFromBool(bool flag) {
  return flag ? Override::kOn : Override::kOff;
}

static inline bool ApplyOverride(Override o, bool condition) {
  if (o == Override::kOn) condition = true;
  if (o == Override::kOff) condition = false;
  return condition;
}

// Additional smoothing helps for medium/low-quality.
enum class GaborishStrength : uint32_t {
  // Serialized, do not change enumerator values.
  kOff = 0,
  k500,
  k750,
  k875,
  k1000,

  // Future extensions: [5, 6]
  kMaxValue
};

struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = 1.0f;
  size_t target_size = 0;
  float target_bitrate = 0.0f;

  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  float quant_border_bias = 0.0f;

  // If true, will use a compression method that is reasonably fast and aims to
  // find a trade-off between quality and file size that optimizes the
  // quality-adjusted-bits-per-pixel metric.
  bool fast_mode = false;
  int max_butteraugli_iters = 11;

  bool guetzli_mode = false;
  int max_butteraugli_iters_guetzli_mode = 100;

  bool lossless_mode = false;

  Override noise = Override::kDefault;
  Override gradient = Override::kDefault;

  Override adaptive_reconstruction = Override::kDefault;
  // Optional parameters for adaptive reconstruction.
  Override epf_use_sharpened = Override::kDefault;
  uint32_t epf_sigma = 0;  // 0 means adaptive

  int gaborish = int(GaborishStrength::k750);

  bool use_ac_strategy = false;

  // Progressive mode.
  bool progressive_mode = false;

  // Progressive-mode saliency extractor.
  // Empty string disables this feature.
  std::string saliency_extractor_for_progressive_mode;
  std::string xclbinPath;
  // Every saliency-heatmap cell with saliency > threshold will be considered as
  // 'salient'.
  float saliency_threshold = 1.0f;
  // Debug parameter: If true, drop non-salient AC part in progressive encoding.
  bool saliency_debug_skip_nonsalient = false;

  // Input and output file name. Will be used to provide pluggable saliency
  // extractor with paths.
  const char* file_in = nullptr;
  const char* file_out = nullptr;

  // Whether to keep temporary files (used e.g. to communicate with external
  // saliency extractor).
  bool keep_tempfiles = false;

  // Prints extra information during/after encoding.
  bool verbose = false;

  // Multiplier for penalizing new HF artifacts more than blurring away
  // features. 1.0=neutral.
  float hf_asymmetry = 1.0f;

  // Intended intensity target of the viewer after decoding, in nits (cd/m^2).
  // There is no other way of knowing the target brightness - depends on source
  // material. 709 typically targets 100 nits, 2020 PQ up to 10K, but HDR
  // content is more typically mastered to 4K nits. The default requires no
  // scaling for Butteraugli.
  float intensity_target = kDefaultIntensityTarget;

  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  // Enable LF/HF predictions.
  bool predict_lf = false;
  bool predict_hf = false;

  float GetIntensityMultiplier() const {
    return intensity_target * kIntensityMultiplier;
  }
};

struct DecompressParams {
  uint64_t max_num_pixels = (1 << 30) - 1;
  // If true, checks at the end of decoding that all of the compressed data
  // was consumed by the decoder.
  bool check_decompressed_size = true;

  Override noise = Override::kDefault;     // cannot be kOn (needs encoder)
  Override gradient = Override::kDefault;  // cannot be kOn (needs encoder)

  Override adaptive_reconstruction = Override::kDefault;
  // Optional parameters for adaptive reconstruction.
  Override epf_use_sharpened = Override::kDefault;
  uint32_t epf_sigma = 0;  // 0 means adaptive

  int gaborish = -1;

  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  // How many passes to decode at most. By default, decode everything.
  uint32_t max_passes = std::numeric_limits<uint32_t>::max();
  // Alternatively, one can specify the maximum tolerable downscaling factor
  // with respect to the full size of the image. By default, nothing less than
  // the full size is requested.
  size_t max_downsampling = 1;
};

// Enable features for distances >= these thresholds:
static constexpr float kMinButteraugliForNoise = 99.0f;  // disabled
static constexpr float kMinButteraugliForGradient = 99.0f;  // disabled
static constexpr float kMinButteraugliForAdaptiveReconstruction = 0.0f;

}  // namespace pik

#endif  // PIK_PIK_PARAMS_H_
