// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/headers.h"

#include "pik/fields.h"
#include "pik/profiler.h"

namespace pik {

Alpha::Alpha() { Bundle::Init(this); }
ProjectiveTransformParams::ProjectiveTransformParams() { Bundle::Init(this); }
TileHeader::TileHeader() { Bundle::Init(this); }
GroupHeader::GroupHeader() { Bundle::Init(this); }
FrameInfo::FrameInfo() { Bundle::Init(this); }
FrameHeader::FrameHeader() { Bundle::Init(this); }
Preview::Preview() { Bundle::Init(this); }
Animation::Animation() { Bundle::Init(this); }
FileHeader::FileHeader() { Bundle::Init(this); }

Status CanEncode(const TileHeader& tile, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits) {
  return Bundle::CanEncode(tile, extension_bits, total_bits);
}
Status CanEncode(const GroupHeader& group, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits) {
  return Bundle::CanEncode(group, extension_bits, total_bits);
}
Status CanEncode(const FrameHeader& pass, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits) {
  return Bundle::CanEncode(pass, extension_bits, total_bits);
}
Status CanEncode(const FileHeader& file, size_t* PIK_RESTRICT extension_bits,
                 size_t* PIK_RESTRICT total_bits) {
  return Bundle::CanEncode(file, extension_bits, total_bits);
}

Status ReadTileHeader(BitReader* PIK_RESTRICT reader,
                      TileHeader* PIK_RESTRICT tile) {
  PROFILER_FUNC;
  return Bundle::Read(reader, tile);
}
Status ReadGroupHeader(BitReader* PIK_RESTRICT reader,
                       GroupHeader* PIK_RESTRICT group) {
  PROFILER_FUNC;
  return Bundle::Read(reader, group);
}
Status ReadPassHeader(BitReader* PIK_RESTRICT reader,
                      FrameHeader* PIK_RESTRICT pass) {
  PROFILER_FUNC;
  return Bundle::Read(reader, pass);
}
Status ReadFileHeader(BitReader* PIK_RESTRICT reader,
                      FileHeader* PIK_RESTRICT file) {
  PROFILER_FUNC;
  return Bundle::Read(reader, file);
}

Status WriteTileHeader(const TileHeader& tile, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage) {
  return Bundle::Write(tile, extension_bits, pos, storage);
}
Status WriteGroupHeader(const GroupHeader& group, size_t extension_bits,
                        size_t* PIK_RESTRICT pos, uint8_t* storage) {
  return Bundle::Write(group, extension_bits, pos, storage);
}
Status WritePassHeader(const FrameHeader& pass, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage) {
  return Bundle::Write(pass, extension_bits, pos, storage);
}
Status WriteFileHeader(const FileHeader& file, size_t extension_bits,
                       size_t* PIK_RESTRICT pos, uint8_t* storage) {
  return Bundle::Write(file, extension_bits, pos, storage);
}

void MakeFileHeader(const CompressParams& cparams, const CodecInOut* io,
                    FileHeader* out) {
  out->xsize_minus_1 = io->xsize() - 1;
  out->ysize_minus_1 = io->ysize() - 1;
  Metadata& metadata = out->metadata;
  metadata = io->metadata;
  metadata.target_nits_div50 = cparams.intensity_target / 50;
  metadata.transcoded.original_bit_depth = io->original_bits_per_sample();
  metadata.transcoded.original_color_encoding = io->dec_c_original;
  metadata.transcoded.original_bytes_per_alpha =
      io->HasAlpha() ? io->AlphaBits() / 8 : 0;
  (void)ColorManagement::MaybeRemoveProfile(
      &metadata.transcoded.original_color_encoding);
}

}  // namespace pik
