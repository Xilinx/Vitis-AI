// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COLOR_MANAGEMENT_H_
#define PIK_COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.

#include <stdint.h>
#include <memory>
#include <vector>

#include "pik/color_encoding.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/padded_bytes.h"
#include "pik/status.h"

namespace pik {

// Thread-safe monostate.
struct ColorManagement {
  // Returns false without changing "c" if pp.color_space is unsupported or
  // pp.gamma is outside (0, 1]. Otherwise, sets fields AND c->icc. Used by
  // codecs that provide their own non-ICC metadata.
  static Status SetFromParams(const ProfileParams& pp, ColorEncoding* c);

  // Returns false without changing "c" if "icc" is invalid. Otherwise, sets
  // fields AND c->icc. Used by image codecs that read embedded ICC profiles.
  static Status SetFromProfile(PaddedBytes&& icc, ColorEncoding* c);

  // Returns true and clears c->icc if a subsequent SetProfileFromFields
  // will generate an equivalent profile. If so, there is no need to send the
  // (large) profile in the bitstream.
  static Status MaybeRemoveProfile(ColorEncoding* c);

  // Returns true if c->icc was successfully reconstructed from other fields.
  // This re-establishes the invariant (broken by MaybeRemoveProfile or changing
  // fields) that fields and c->icc are equivalent. Returning false indicates
  // the profile is lost/empty, which means ColorSpaceTransform will fail.
  static Status SetProfileFromFields(ColorEncoding* c);
};

// Run is thread-safe.
class ColorSpaceTransform {
 public:
  ColorSpaceTransform() {}
  ~ColorSpaceTransform();

  // Cannot copy (transforms_ holds pointers).
  ColorSpaceTransform(const ColorSpaceTransform&) = delete;
  ColorSpaceTransform& operator=(const ColorSpaceTransform&) = delete;

  // "Constructor"; allocates for up to `num_threads`, or returns false.
  Status Init(const ColorEncoding& c_src, const ColorEncoding& c_dst,
              size_t xsize, size_t num_threads);

  float* PIK_RESTRICT BufSrc(const size_t thread) {
    return buf_src_.Row(thread);
  }

  float* PIK_RESTRICT BufDst(const size_t thread) {
    return buf_dst_.Row(thread);
  }

  // buf_X can either be from BufX() or caller-allocated, interleaved storage.
  // `thread` must be less than the `num_threads` passed to Init.
  void Run(const size_t thread, const float* buf_src, float* buf_dst);

 private:
  enum class ExtraTF {
    kNone,
    kPQ,
    kHLG,
    kSRGB,
  };

  // One per thread - cannot share because of caching.
  std::vector<void*> transforms_;

  ImageF buf_src_;
  ImageF buf_dst_;
  size_t xsize_;
  bool skip_lcms_ = false;
  ExtraTF preprocess_ = ExtraTF::kNone;
  ExtraTF postprocess_ = ExtraTF::kNone;
};

}  // namespace pik

#endif  // PIK_COLOR_MANAGEMENT_H_
