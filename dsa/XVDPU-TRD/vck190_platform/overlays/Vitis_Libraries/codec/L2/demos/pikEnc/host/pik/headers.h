// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_HEADERS_H_
#define PIK_HEADERS_H_

// Group/pass/file headers with backward and forward-compatible extension
// capability and compressed integer fields.

#include <stddef.h>
#include <stdint.h>

#include "pik/bit_reader.h"
#include "pik/codec.h"
#include "pik/color_encoding.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/epf.h"
#include "pik/field_encodings.h"
#include "pik/gaborish.h"
#include "pik/metadata.h"
#include "pik/padded_bytes.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

//------------------------------------------------------------------------------
// Tile

constexpr size_t kNumProjectiveTransformParams = 8;

struct ProjectiveTransformParams {
  ProjectiveTransformParams();
  static const char* Name() { return "ProjectiveTransformParams"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    for (size_t i = 0; i < kNumProjectiveTransformParams; ++i) {
      visitor->U32(kU32RawBits + 8, 1, &corner_coords[i]);
    }

    return true;
  }

  uint32_t corner_coords[kNumProjectiveTransformParams];
};

struct TileHeader {
  TileHeader();
  static const char* Name() { return "TileHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->Bool(false, &have_projective_transform);
    if (visitor->Conditional(have_projective_transform)) {
      PIK_RETURN_IF_ERROR(visitor->VisitNested(&projective_transform_params));
    }

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  bool all_default;

  bool have_projective_transform;
  ProjectiveTransformParams projective_transform_params;

  uint64_t extensions;
};

//------------------------------------------------------------------------------
// Group

// Alpha channel (lossless compression).
// TODO(janwas): add analogous depth-image support
struct Alpha {
  Alpha();
  static const char* Name() { return "Alpha"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    // TODO(janwas): use this instead of have_alpha
    // if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->U32(0x84828180u, 1, &bytes_per_alpha);
    visitor->Bytes(BytesEncoding::kRaw, &encoded);

    return true;
  }

  // TODO(b/120660058): Move bytes_per_alpha to container.
  uint32_t bytes_per_alpha;
  PaddedBytes encoded;
};

constexpr size_t kNumTilesPerGroup = (kGroupDimInBlocks / kTileDimInBlocks) *
                                     (kGroupDimInBlocks / kTileDimInBlocks);

struct GroupHeader {
  GroupHeader();
  static const char* Name() { return "GroupHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    if (visitor->Conditional(nonserialized_have_alpha)) {
      PIK_RETURN_IF_ERROR(visitor->VisitNested(&alpha));
    }

    // TODO(user): Skip all tiles if all of them are all_default.
    for (size_t i = 0; i < kNumTilesPerGroup; ++i) {
      PIK_RETURN_IF_ERROR(visitor->VisitNested(&tile_headers[i]));
    }

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  bool all_default;

  bool nonserialized_have_alpha = false;
  Alpha alpha;

  TileHeader tile_headers[kNumTilesPerGroup];

  uint64_t extensions;
};

//------------------------------------------------------------------------------
// Frame

enum class ImageEncoding : uint32_t {
  kPasses = 0,   // PIK
  kProgressive,  // FUIF
  kLossless,
  // TODO(lode): extend amount of possible values
  // Future extensions: [6]
};

struct FrameInfo {
  FrameInfo();
  static const char* Name() { return "FrameInfo"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->U32(0x20088180, 0, &duration);

    visitor->Bool(false, &have_timecode);
    if (visitor->Conditional(have_timecode)) {
      visitor->U32(kU32RawBits + 32, 0, &timecode);
    }

    visitor->Bool(false, &is_keyframe);

    return true;
  }

  bool all_default;

  // How long to wait [in ticks, see Animation{}] after rendering
  uint32_t duration;

  bool have_timecode;
  uint32_t timecode;  // 0xHHMMSSFF

  bool is_keyframe;
};

// Image/frame := one of more of these, where the last has is_last = true.
// Starts at a byte-aligned address "a"; the next pass starts at "a + size".
struct FrameHeader {
  // Optional postprocessing steps. These flags are the source of truth;
  // Override must set/clear them rather than change their meaning.
  enum Flags {
    // Gradient map used to predict smooth areas.
    kGradientMap = 1,

    // Image is compressed with grayscale optimizations. Only used for parsing
    // of pik file, may not be used to determine decompressed color format or
    // ICC color profile.
    kGrayscaleOpt = 2,

    // Inject noise into decoded output.
    kNoise = 4,
  };

  FrameHeader();
  static const char* Name() { return "FrameHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->U64(0, &size);

    visitor->Bool(false, &has_alpha);
    PIK_RETURN_IF_ERROR(visitor->VisitNested(&frame));

    visitor->Enum(kU32Direct3Plus4, ImageEncoding::kPasses, &encoding);

    // Flags, AC strategy, AR and predictions only make sense for kPasses.
    if (visitor->Conditional(encoding == ImageEncoding::kPasses)) {
      visitor->U32(0x20181008, 0, &flags);
      visitor->Enum(kU32Direct3Plus4, GaborishStrength::k750, &gaborish);

      visitor->Bool(true, &predict_lf);
      visitor->Bool(true, &predict_hf);
      visitor->Bool(false, &have_adaptive_reconstruction);
      if (visitor->Conditional(have_adaptive_reconstruction)) {
        PIK_RETURN_IF_ERROR(visitor->VisitNested(&epf_params));
      }
      // TODO(veluca); choose a good constant.
      visitor->U32(0x20181008, 1, &num_passes);

      uint32_t num_downsampling_factors = downsampling_factor_to_passes.size();
      visitor->U32(kU32Direct0To3, 0, &num_downsampling_factors);
      visitor->SetSizeWhenReading(num_downsampling_factors,
                                  &downsampling_factor_to_passes);
      for (auto& downsampling_and_num_passes : downsampling_factor_to_passes) {
        visitor->U32(kU32Direct1248, 1, &downsampling_and_num_passes.first);
        visitor->U32(kU32Direct3Plus8, 1, &downsampling_and_num_passes.second);
      }
    }

    // No resampling or group TOC for kProgressive.
    if (visitor->Conditional(encoding != ImageEncoding::kProgressive)) {
      // WARNING: nonserialized_num_groups must be set beforehand.
      visitor->SetSizeWhenReading(nonserialized_num_groups, &group_sizes);
      for (uint32_t& group_size_bits : group_sizes) {
        visitor->U32(0x150F0E0C, 0, &group_size_bits);
      }
    }

    if (visitor->Conditional(encoding == ImageEncoding::kLossless)) {
      visitor->Bool(false, &lossless_grayscale);
      visitor->Bool(false, &lossless_16_bits);
    }

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  // Relative to START of (byte-aligned) FrameHeader. Used to seek to next
  // frame.
  // TODO(veluca): how do we compute this?
  uint64_t size;  // [bytes]
  bool has_alpha;

  FrameInfo frame;

  ImageEncoding encoding;

  // Lossless encoding flags: grayscale mode, 16 (true) or 8 bit (false) mode.
  bool lossless_grayscale;
  bool lossless_16_bits;

  uint32_t flags;

  GaborishStrength gaborish;

  bool predict_lf;
  bool predict_hf;

  // TODO(janwas): move into EpfParams
  bool have_adaptive_reconstruction;
  EpfParams epf_params;

  uint32_t num_passes;

  // WARNING: must be set before reading from bitstream - not serialized
  // like other fields because this is stored in FileHeader to save a few bits.
  size_t nonserialized_num_groups = 0;

  std::vector<uint32_t> group_sizes;  // TOC, [bytes]

  // Pairs of {max downsampling factor, number of passes to decode}.
  // It is not necessary to include {1, num_passes} or {8, 0} explicitly. Code
  // that uses this vector always behaves as though they are there.
  std::vector<std::pair<uint32_t, uint32_t>> downsampling_factor_to_passes;

  // TODO(janwas): quantization setup (reuse from previous passes)

  uint64_t extensions;
};

//------------------------------------------------------------------------------
// File

struct Preview {
  Preview();
  static const char* Name() { return "Preview"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->U32(0x1C14100C, 0, &size_bits);
    visitor->U32(0x0D0B0907, 0, &xsize);
    visitor->U32(0x0D0B0907, 0, &ysize);

    return true;
  }

  bool all_default;

  uint32_t size_bits;
  uint32_t xsize;
  uint32_t ysize;
};

struct Animation {
  Animation();
  static const char* Name() { return "Animation"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->U32(0x20100380, 0, &num_loops);
    visitor->U32(0x20140981, 0, &ticks_numerator);
    visitor->U32(0x20140981, 1, &ticks_denominator);

    return true;
  }

  bool all_default;

  uint32_t num_loops;  // 0 means to repeat infinitely.

  // Ticks as rational number in seconds per tick
  uint32_t ticks_numerator;
  uint32_t ticks_denominator;  // Must be at least 1
};

// EXIF orientation of the image. This field overrides any field present in
// actual EXIF metadata. The value tells which transformation the decoder must
// apply after decoding to display the image with the correct orientation.
enum class Orientation : uint32_t {
  kIdentity = 1,
  kFlipHorizontal = 2,
  kRotate180 = 3,
  kFlipVertical = 4,
  kTranspose = 5,
  kRotate90 = 6,
  kAntiTranspose = 7,
  kRotate270 = 8,
};

// Followed by an unbounded stream of interleaved FrameHeader+payloads.
struct FileHeader {
  FileHeader();
  static const char* Name() { return "FileHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->U32(kU32RawBits + 32, kSignature, &signature);
    if (signature != kSignature) return PIK_FAILURE("Signature mismatch");

    // Almost all camera images are less than 8K * 8K. We also allow the
    // full 32-bit range for completeness.
    visitor->U32(0x200D0B09, 0, &xsize_minus_1);
    visitor->U32(0x200D0B09, 0, &ysize_minus_1);
    visitor->Orientation(Orientation::kIdentity, &orientation);

    PIK_RETURN_IF_ERROR(visitor->VisitNested(&metadata));
    PIK_RETURN_IF_ERROR(visitor->VisitNested(&preview));
    PIK_RETURN_IF_ERROR(visitor->VisitNested(&animation));

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  size_t xsize() const { return xsize_minus_1 + 1; }
  size_t ysize() const { return ysize_minus_1 + 1; }

  // \n causes files opened in text mode to be rejected, and \xD7 detects
  // 7-bit transfers (it also looks like x in ISO-8859-1).
  static constexpr uint32_t kSignature = 0x0A4D4CD7;  // xLM\n
  uint32_t signature;

  // This encoding saves bits for size=8K and prevents invalid size=0.
  uint32_t xsize_minus_1;
  uint32_t ysize_minus_1;

  Orientation orientation;

  Metadata metadata;
  Preview preview;
  Animation animation;

  uint64_t extensions;
};

void MakeFileHeader(const CompressParams& cparams, const CodecInOut* io,
                    FileHeader* out);

// Returns whether a header's fields can all be encoded, i.e. they have a valid
// representation. If so, "*total_bits" is the exact number of bits required.
Status CanEncode(const TileHeader& tile, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits);
Status CanEncode(const GroupHeader& group, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits);
Status CanEncode(const FrameHeader& pass, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits);
Status CanEncode(const FileHeader& file, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits);

Status ReadTileHeader(BitReader* PIK_RESTRICT reader,
                      TileHeader* PIK_RESTRICT tile);
Status ReadGroupHeader(BitReader* PIK_RESTRICT reader,
                       GroupHeader* PIK_RESTRICT group);
Status ReadPassHeader(BitReader* PIK_RESTRICT reader,
                      FrameHeader* PIK_RESTRICT pass);
Status ReadFileHeader(BitReader* PIK_RESTRICT reader,
                      FileHeader* PIK_RESTRICT file);

// "extension_bits" is from the preceding call to CanEncode.
Status WriteTileHeader(const TileHeader& tile, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage);
Status WriteGroupHeader(const GroupHeader& group, size_t extension_bits,
                        size_t* PIK_RESTRICT pos, uint8_t* storage);
Status WritePassHeader(const FrameHeader& pass, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage);
Status WriteFileHeader(const FileHeader& file, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage);

}  // namespace pik

#endif  // PIK_HEADERS_H_
