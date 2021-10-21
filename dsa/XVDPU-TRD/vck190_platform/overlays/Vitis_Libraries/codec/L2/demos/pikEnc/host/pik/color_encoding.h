// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COLOR_ENCODING_H_
#define PIK_COLOR_ENCODING_H_

// Metadata for color space conversions.

#include <stddef.h>
#include <stdint.h>
#include <cmath>  // std::abs
#include <string>
#include <vector>

#include "pik/compiler_specific.h"
#include "pik/field_encodings.h"
#include "pik/padded_bytes.h"
#include "pik/status.h"

namespace pik {

// (All CIE units are for the standard 1931 2 degree observer)

enum class ColorSpace : uint32_t {
  kRGB = 0,
  kGray,
  kXYZ,
  kUnknown,
  kYCbCr,  // from BT.2100
  kICtCp,  // from BT.2100
  // Future extensions: [6, 10]
};

enum class WhitePoint : uint32_t {
  kD65 = 0,  // sRGB/BT.709/P3/BT.2020/Adobe
  kD60,      // ACES
  kD50,      // ICC PCS
  kUnknown,
  kE,  // XYZ
  // Future extensions: [5, 10]
};

enum class Primaries : uint32_t {
  kSRGB = 0,  // Same as BT.709
  k2020,      // Same as BT.2100
  kP3,
  kUnknown,
  kAP0,  // from ACES
  kAP1,  // from ACEScc/ACEScg
  kAdobe,
  // Future extensions: [7, 10]
};

enum TransferFunction : uint32_t {
  kSRGB = 0,
  kLinear,
  kPQ,  // from BT.2100
  kUnknown,
  k709,
  kAdobe,
  kHLG,  // from BT.2100
  // Future extensions: [7, 10]
};

enum class RenderingIntent : uint32_t {
  // Values match ICC sRGB encodings.
  kPerceptual = 0,  // good for photos, requires a profile with LUT.
  kRelative,        // good for logos.
  kSaturation,      // perhaps useful for CG with fully saturated colors.
  kAbsolute,        // leaves white point unchanged; good for proofing.
  kUnknown          // invalid, only used for parsing
  // Future extensions: [5, 10]
};

// For generating profile descriptions.
std::string ToString(ColorSpace color_space);
std::string ToString(WhitePoint white_point);
std::string ToString(Primaries primaries);
std::string ToString(TransferFunction transfer_function);
std::string ToString(RenderingIntent rendering_intent);

// Used by AllEncodings and ParseDescription.
std::vector<ColorSpace> Values(ColorSpace);
std::vector<WhitePoint> Values(WhitePoint);
std::vector<Primaries> Values(Primaries);
std::vector<TransferFunction> Values(TransferFunction);
std::vector<RenderingIntent> Values(RenderingIntent);

// Convenience wrapper: takes care of passing ::kUnknown.
template <typename Enum>
std::vector<Enum> AllValues() {
  return Values(Enum::kUnknown);
}

// Chromaticity (Y is omitted because it is 1 for primaries/white points)
struct CIExy {
  double x = 0.0;
  double y = 0.0;
};

struct PrimariesCIExy {
  CIExy r;
  CIExy g;
  CIExy b;
};

WhitePoint WhitePointFromCIExy(const CIExy& xy);
// Returns false if white_point == kUnknown.
Status WhitePointToCIExy(WhitePoint white_point, CIExy* PIK_RESTRICT xy);

Primaries PrimariesFromCIExy(const PrimariesCIExy& xy);
// Returns false if primaries == kUnknown.
Status PrimariesToCIExy(Primaries primaries, PrimariesCIExy* PIK_RESTRICT xy);

static inline bool IsLinear(const TransferFunction tf) {
  return tf == TransferFunction::kLinear;
}

static inline bool IsSRGB(const TransferFunction tf) {
  return tf == TransferFunction::kSRGB;
}

static inline bool IsPQ(const TransferFunction tf) {
  return tf == TransferFunction::kPQ;
}

static inline bool Is2100(const TransferFunction tf) {
  return tf == TransferFunction::kPQ || tf == TransferFunction::kHLG;
}

// All data required to interpret and translate pixels to a known color space.
// For most images (i.e. those with a known ICC profile), the encoded size is
// only 10 bits. Stored in Metadata.
struct ColorEncoding {
  ColorEncoding();
  static const char* Name() { return "ColorEncoding"; }

  bool IsGray() const { return color_space == ColorSpace::kGray; }
  size_t Channels() const { return IsGray() ? 1 : 3; }

  bool IsSRGB() const {
    return white_point == WhitePoint::kD65 && primaries == Primaries::kSRGB &&
           pik::IsSRGB(transfer_function);
  }

  void SetSRGB(const ColorSpace cs) {
    icc.clear();
    PIK_ASSERT(cs == ColorSpace::kGray || cs == ColorSpace::kRGB)
    color_space = cs;
    white_point = WhitePoint::kD65;
    primaries = Primaries::kSRGB;
    transfer_function = TransferFunction::kSRGB;
    rendering_intent = RenderingIntent::kPerceptual;
  }

  template <class Visitor>
  bool VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->Bytes(BytesEncoding::kBrotli, &icc);

    visitor->Enum(kU32Direct3Plus8, ColorSpace::kRGB, &color_space);
    visitor->Enum(kU32Direct3Plus8, WhitePoint::kD65, &white_point);
    visitor->Enum(kU32Direct3Plus8, Primaries::kSRGB, &primaries);
    visitor->Enum(kU32Direct3Plus8, TransferFunction::kSRGB,
                  &transfer_function);
    visitor->Enum(kU32Direct3Plus8, RenderingIntent::kPerceptual,
                  &rendering_intent);

    return true;
  }

  // The enum fields should always describe attributes of "icc" except:
  // - between MaybeRemoveProfile and SetProfileFromFields (icc empty);
  // - between ctor/setting fields and SetProfileFromFields (icc empty);
  // - after SetFromProfile of an unusual profile (fields may be kUnknown).
  PaddedBytes icc;
  ColorSpace color_space;
  WhitePoint white_point;  // unused if kXYZ
  Primaries primaries;     // unused if kGray or kXYZ
  TransferFunction transfer_function;
  RenderingIntent rendering_intent;

  bool SameColorSpace(const ColorEncoding& other) const {
    if (color_space != other.color_space) return false;
    if (color_space == ColorSpace::kXYZ) return true;
    if (white_point != other.white_point) return false;
    if (color_space == ColorSpace::kGray) return true;
    return primaries == other.primaries;
  }
};

// Returns whether the two inputs are approximately equal.
static inline bool ApproxEq(const double a, const double b) {
  // Threshold is sufficient for ICC's 15-bit fixed-point numbers.
  return std::abs(a - b) <= 6E-5;
}

// Floating-point "gamma" is an alternative to TransferFunction that allows
// other codecs to specify arbitrary exponents.

// All return values except Linear are arbitrary and only useful for comparison.
static inline constexpr double GammaUnknown() { return 0.0; }
static inline constexpr double GammaLinear() { return 1.0; }
static inline constexpr double GammaSRGB() { return 1.0 / 2.2; }
static inline constexpr double GammaAdobe() { return 1.0 / 2.19921875; }
static inline constexpr double Gamma709() { return 1.0 / 2.0; }
static inline constexpr double GammaPQ() { return 0.15; }
static inline constexpr double GammaHLG() { return 0.125; }

double GammaFromTransferFunction(TransferFunction tf);  // Returns Gamma*().
TransferFunction TransferFunctionFromGamma(double gamma);

// For Description.
std::string StringFromWhitePoint(const CIExy& xy);
std::string StringFromPrimaries(const PrimariesCIExy& xy);
std::string StringFromGamma(double gamma);
double GammaFromString(const std::string& s);

// Sufficient information to create an ICC profile. Used by other image codecs
// as an alternative to embedding ICC. Same fields as ColorEncoding, but allows
// arbitrary white point/primaries/gamma.
struct ProfileParams {
  ColorSpace color_space;
  CIExy white_point;         // ignored if kXYZ
  PrimariesCIExy primaries;  // ignored if kGray or kXYZ
  double gamma;
  RenderingIntent rendering_intent;
};

// Example: "RGB_D65_SRG_Rel_Lin"
std::string Description(const ColorEncoding& c);  // from fields, not icc
std::string Description(const ProfileParams& pp);
Status ParseDescription(const std::string& description,
                        ProfileParams* PIK_RESTRICT pp);

Status ColorEncodingToParams(const ColorEncoding& c,
                             ProfileParams* PIK_RESTRICT pp);
void SetFieldsFromParams(const ProfileParams& pp,
                         ColorEncoding* PIK_RESTRICT c);

// Returns ColorEncoding with empty ICC profile. Caller must use
// ColorEncoding::SetProfileFromFields() to generate a profile.
std::vector<ColorEncoding> AllEncodings();

}  // namespace pik

#endif  // PIK_COLOR_ENCODING_H_
