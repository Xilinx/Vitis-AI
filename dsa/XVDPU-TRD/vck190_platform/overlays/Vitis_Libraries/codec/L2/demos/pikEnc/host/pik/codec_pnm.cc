// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/codec_pnm.h"

#include <string>

#include "pik/bits.h"
#include "pik/byte_order.h"
#include "pik/external_image.h"
#include "pik/fields.h"

namespace pik {
namespace {

struct HeaderPNM {
  size_t xsize;
  size_t ysize;
  bool is_gray;
  size_t bits_per_sample;
  bool big_endian;
};

class Parser {
 public:
  explicit Parser(const PaddedBytes& bytes)
      : pos_(bytes.data()), end_(pos_ + bytes.size()) {}

  // Sets "pos" to the first non-header byte/pixel on success.
  Status ParseHeader(HeaderPNM* header, const uint8_t** pos) {
    // codec_facade ensures we have at least two bytes => no range check here.
    if (pos_[0] != 'P') return false;
    const uint8_t type = pos_[1];
    pos_ += 2;

    switch (type) {
      case '5':
        header->is_gray = true;
        return ParseHeaderPNM(header, pos);

      case '6':
        header->is_gray = false;
        return ParseHeaderPNM(header, pos);

      case 'F':
        header->is_gray = false;
        return ParseHeaderPFM(header, pos);

      case 'f':
        header->is_gray = true;
        return ParseHeaderPFM(header, pos);
    }
    return false;
  }

 private:
  static bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  Status ParseUnsigned(size_t* number) {
    if (pos_ == end_) return PIK_FAILURE("PNM: reached end before number");
    if (!IsDigit(*pos_)) return PIK_FAILURE("PNM: expected unsigned number");

    *number = 0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    return true;
  }

  Status ParseSigned(double* number) {
    if (pos_ == end_) return PIK_FAILURE("PNM: reached end before number");
    if (*pos_ != '-' && *pos_ != '+' && !IsDigit(*pos_)) {
      return PIK_FAILURE("PNM: expected signed number");
    }

    const size_t max_size = std::min<ptrdiff_t>(end_ - pos_, 50);
    const std::string copy(reinterpret_cast<const char*>(pos_), max_size);
    size_t chars_processed;
    *number = std::stod(copy, &chars_processed);
    pos_ += chars_processed;

    return true;
  }

  Status SkipBlank() {
    if (pos_ == end_) return PIK_FAILURE("PNM: reached end before blank");
    if (*pos_ != ' ') return PIK_FAILURE("PNM: expected blank");
    ++pos_;
    return true;
  }

  Status SkipSingleWhitespace() {
    if (pos_ == end_) return PIK_FAILURE("PNM: reached end before whitespace");
    if (!IsWhitespace(*pos_)) return PIK_FAILURE("PNM: expected whitespace");
    ++pos_;
    return true;
  }

  Status SkipWhitespace() {
    if (pos_ == end_) return PIK_FAILURE("PNM: reached end before whitespace");
    if (!IsWhitespace(*pos_) && *pos_ != '#') {
      return PIK_FAILURE("PNM: expected whitespace/comment");
    }

    while (pos_ < end_ && IsWhitespace(*pos_)) {
      ++pos_;
    }

    // Comment(s)
    while (pos_ != end_ && *pos_ == '#') {
      while (pos_ != end_ && !IsLineBreak(*pos_)) {
        ++pos_;
      }
      // Newline(s)
      while (pos_ != end_ && IsLineBreak(*pos_)) pos_++;
    }

    while (pos_ < end_ && IsWhitespace(*pos_)) {
      ++pos_;
    }
    return true;
  }

  Status ParseHeaderPNM(HeaderPNM* header, const uint8_t** pos) {
    PIK_RETURN_IF_ERROR(SkipWhitespace());
    PIK_RETURN_IF_ERROR(ParseUnsigned(&header->xsize));

    PIK_RETURN_IF_ERROR(SkipWhitespace());
    PIK_RETURN_IF_ERROR(ParseUnsigned(&header->ysize));

    PIK_RETURN_IF_ERROR(SkipWhitespace());
    size_t max_val;
    PIK_RETURN_IF_ERROR(ParseUnsigned(&max_val));
    if (max_val == 0 || max_val >= 65536) return PIK_FAILURE("PNM: bad MaxVal");
    header->bits_per_sample = CeilLog2Nonzero(static_cast<uint32_t>(max_val));
    header->big_endian = true;

    PIK_RETURN_IF_ERROR(SkipSingleWhitespace());

    *pos = pos_;
    return true;
  }

  Status ParseHeaderPFM(HeaderPNM* header, const uint8_t** pos) {
    PIK_RETURN_IF_ERROR(SkipSingleWhitespace());
    PIK_RETURN_IF_ERROR(ParseUnsigned(&header->xsize));

    PIK_RETURN_IF_ERROR(SkipBlank());
    PIK_RETURN_IF_ERROR(ParseUnsigned(&header->ysize));

    PIK_RETURN_IF_ERROR(SkipSingleWhitespace());
    double scale;
    PIK_RETURN_IF_ERROR(ParseSigned(&scale));
    header->big_endian = scale >= 0.0;
    header->bits_per_sample = 32;

    PIK_RETURN_IF_ERROR(SkipSingleWhitespace());

    *pos = pos_;
    return true;
  }

  const uint8_t* pos_;
  const uint8_t* const end_;
};

constexpr size_t kMaxHeaderSize = 200;

Status EncodeHeader(const ExternalImage& external, char* header,
                    int* PIK_RESTRICT chars_written) {
  if (external.HasAlpha()) return PIK_FAILURE("PNM: can't store alpha");

  if (external.BitsPerSample() == 32) {  // PFM
    const char type = external.IsGray() ? 'f' : 'F';
    const double scale = external.BigEndian() ? 1.0 : -1.0;
    snprintf(header, kMaxHeaderSize, "P%c %zu %zu\n%f\n%n", type,
             external.xsize(), external.ysize(), scale, chars_written);
  } else {  // PGM/PPM
    const uint32_t max_val = (1U << external.BitsPerSample()) - 1;
    if (max_val >= 65536) return PIK_FAILURE("PNM cannot have > 16 bits");
    const char type = external.IsGray() ? '5' : '6';
    snprintf(header, kMaxHeaderSize, "P%c\n%zu %zu\n%u\n%n", type,
             external.xsize(), external.ysize(), max_val, chars_written);
  }
  return true;
}

Status ApplyHints(const bool is_gray, CodecInOut* io) {
  bool got_color_space = false;
  Status ok = true;

  io->dec_hints.Foreach([is_gray, io, &got_color_space, &ok](
                            const std::string& key, const std::string& value) {
    if (key == "color_space") {
      ProfileParams pp;
      if (!ParseDescription(value, &pp) ||
          !ColorManagement::SetFromParams(pp, &io->dec_c_original)) {
        fprintf(stderr, "PNM: Failed to apply color_space.\n");
        ok = false;
      }

      if (is_gray != io->dec_c_original.IsGray()) {
        fprintf(stderr, "PNM: mismatch between file and color_space hint.\n");
        ok = false;
      }

      got_color_space = true;
    } else {
      fprintf(stderr, "PNM decoder ignoring %s hint\n", key.c_str());
    }
  });

  if (!got_color_space) {
    fprintf(stderr, "PNM: no color_space hint given, assuming sRGB.\n");
    io->dec_c_original.SetSRGB(is_gray ? ColorSpace::kGray : ColorSpace::kRGB);
    PIK_RETURN_IF_ERROR(
        ColorManagement::SetProfileFromFields(&io->dec_c_original));
  }

  if (!ok) return PIK_FAILURE("PNM ApplyHints failed");
  return true;
}

}  // namespace

Status DecodeImagePNM(const PaddedBytes& bytes, ThreadPool* pool,
                      CodecInOut* io) {
  io->enc_size = bytes.size();

  Parser parser(bytes);
  HeaderPNM header;
  const uint8_t* pos;
  PIK_RETURN_IF_ERROR(parser.ParseHeader(&header, &pos));

  PIK_RETURN_IF_ERROR(ApplyHints(header.is_gray, io));
  io->SetOriginalBitsPerSample(header.bits_per_sample);
  io->metadata = Metadata();

  const bool has_alpha = false;
  const uint8_t* end = bytes.data() + bytes.size();
  const ExternalImage external(header.xsize, header.ysize, io->dec_c_original,
                               has_alpha,  /*alpha_bits=*/ 0,
                               header.bits_per_sample, header.big_endian,
                               pos, end);
  const CodecIntervals* temp_intervals = nullptr;  // Don't know min/max.
  return external.CopyTo(temp_intervals, pool, io);
}

Status EncodeImagePNM(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      PaddedBytes* bytes) {
  io->enc_bits_per_sample = bits_per_sample <= 16 ? bits_per_sample : 32;
  // Choose native for PFM; PGM/PPM require big-endian.
  const bool big_endian = (bits_per_sample == 32) ? !IsLittleEndian() : true;

  if (!Bundle::AllDefault(io->metadata)) {
    fprintf(stderr, "PNM encoder ignoring metadata - use a different codec.\n");
  }
  if (!c_desired.IsSRGB()) {
    fprintf(stderr,
            "PNM encoder cannot store custom ICC profile; decoder "
            "will need hint key=color_space to get the same values.\n");
  }

  const ImageU* alpha = io->HasAlpha() ? &io->alpha() : nullptr;
  const size_t alpha_bits = io->HasAlpha() ? io->AlphaBits() : 0;
  CodecIntervals* temp_intervals = nullptr;  // Can't store min/max.
  ExternalImage external(pool, io->color(), Rect(io->color()), io->c_current(),
                         c_desired, io->HasAlpha(), alpha, alpha_bits,
                         io->enc_bits_per_sample, big_endian, temp_intervals);
  PIK_RETURN_IF_ERROR(external.IsHealthy());

  char header[kMaxHeaderSize];
  int header_size = 0;
  PIK_RETURN_IF_ERROR(EncodeHeader(external, header, &header_size));

  const PaddedBytes& pixels = external.Bytes();
  io->enc_size = header_size + pixels.size();
  bytes->resize(io->enc_size);
  memcpy(bytes->data(), header, header_size);
  memcpy(bytes->data() + header_size, pixels.data(), pixels.size());

  return true;
}

}  // namespace pik
