// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/color_encoding.h"

#include "pik/fields.h"

namespace pik {

// These strings are baked into Description - do not change. Fixed-length
// simplifies parsing. Despite enum class, invalid bitstreams may lead to
// invalid enums, so handle them gracefully.

std::string ToString(ColorSpace color_space) {
  switch (color_space) {
    case ColorSpace::kRGB:
      return "RGB";
    case ColorSpace::kGray:
      return "Gra";
    case ColorSpace::kXYZ:
      return "XYZ";
    case ColorSpace::kUnknown:
      return "Unk";
    case ColorSpace::kYCbCr:
      return "YCC";
    case ColorSpace::kICtCp:
      return "ITP";
  }
  return "CS?";
}

std::string ToString(WhitePoint white_point) {
  switch (white_point) {
    case WhitePoint::kD65:
      return "D65";
    case WhitePoint::kD60:
      return "D60";
    case WhitePoint::kD50:
      return "D50";
    case WhitePoint::kE:
      return "EER";
    case WhitePoint::kUnknown:
      return "Unk";
  }
  return "WP?";
}

std::string ToString(Primaries primaries) {
  switch (primaries) {
    case Primaries::kSRGB:
      return "SRG";
    case Primaries::k2020:
      return "202";
    case Primaries::kP3:
      return "DCI";
    case Primaries::kUnknown:
      return "Unk";
    case Primaries::kAP0:
      return "AP0";
    case Primaries::kAP1:
      return "AP1";
    case Primaries::kAdobe:
      return "Ado";
  }
  return "PR?";
}

std::string ToString(TransferFunction transfer_function) {
  switch (transfer_function) {
    case TransferFunction::kSRGB:
      return "SRG";
    case TransferFunction::kAdobe:
      return "Ado";
    case TransferFunction::kLinear:
      return "Lin";
    case TransferFunction::k709:
      return "709";
    case TransferFunction::kUnknown:
      return "Unk";
    case TransferFunction::kPQ:
      return "PeQ";
    case TransferFunction::kHLG:
      return "HLG";
  }
  return "TF?";
}

std::string ToString(RenderingIntent rendering_intent) {
  switch (rendering_intent) {
    case RenderingIntent::kPerceptual:
      return "Per";
    case RenderingIntent::kRelative:
      return "Rel";
    case RenderingIntent::kSaturation:
      return "Sat";
    case RenderingIntent::kAbsolute:
      return "Abs";
    case RenderingIntent::kUnknown:
      return "Unk";
  }
  return "RI?";
}

// Returns all enumerators (except unknown). Used by ParseString and tests.
// The parameter is only used for type dispatch.

std::vector<ColorSpace> Values(ColorSpace) {
  return {ColorSpace::kRGB, ColorSpace::kGray, ColorSpace::kXYZ,
          ColorSpace::kYCbCr, ColorSpace::kICtCp};
}

std::vector<WhitePoint> Values(WhitePoint) {
  return {WhitePoint::kD65, WhitePoint::kD60, WhitePoint::kD50, WhitePoint::kE};
}

std::vector<Primaries> Values(Primaries) {
  return {Primaries::kSRGB, Primaries::k2020, Primaries::kP3,
          Primaries::kAP0,  Primaries::kAP1,  Primaries::kAdobe};
}

std::vector<TransferFunction> Values(TransferFunction) {
  return {TransferFunction::kSRGB, TransferFunction::kLinear,
          TransferFunction::k709,  TransferFunction::kAdobe,
          TransferFunction::kPQ,   TransferFunction::kHLG};
}

std::vector<RenderingIntent> Values(RenderingIntent) {
  return {RenderingIntent::kPerceptual, RenderingIntent::kRelative,
          RenderingIntent::kSaturation, RenderingIntent::kAbsolute};
}

template <typename Enum>
Enum ValueFromString(const std::string& token) {
  for (Enum e : Values(Enum::kUnknown)) {
    if (ToString(e) == token) return e;
  }
  return Enum::kUnknown;
}

Status WhitePointToCIExy(WhitePoint white_point, CIExy* PIK_RESTRICT xy) {
  switch (white_point) {
    case WhitePoint::kD65:
      // = cmsXYZ2xyY from quantized XYZ = {0.950455927, 1.0, 1.089057751}
      xy->x = 0.312699999963613;
      xy->y = 0.328999999979602;
      return true;

    case WhitePoint::kD60:
      // From https://en.wikipedia.org/wiki/Academy_Color_Encoding_System
      xy->x = 0.32168;
      xy->y = 0.33767;
      return true;

    case WhitePoint::kD50:
      // = cmsXYZ2xyY from quantized XYZ = {0.96420288, 1.0, 0.82490540}
      xy->x = 0.345702921221832;
      xy->y = 0.358537532289711;
      return true;

    case WhitePoint::kE:
      xy->x = xy->y = 1.0 / 3;
      return true;

    case WhitePoint::kUnknown:
      break;  // handled below
  }
  memset(xy, 0, sizeof(*xy));
  return false;
}

WhitePoint WhitePointFromCIExy(const CIExy& xy) {
  if (ApproxEq(xy.x, 0.3127) && ApproxEq(xy.y, 0.3290)) {
    return WhitePoint::kD65;
  }
  if (ApproxEq(xy.x, 0.32168) && ApproxEq(xy.y, 0.33767)) {
    return WhitePoint::kD60;
  }
  if (ApproxEq(xy.x, 0.3457) && ApproxEq(xy.y, 0.3585)) {
    return WhitePoint::kD50;
  }
  if (ApproxEq(xy.x, 1.0 / 3) && ApproxEq(xy.y, 1.0 / 3)) {
    return WhitePoint::kE;
  }

  return WhitePoint::kUnknown;
}

Status PrimariesToCIExy(Primaries primaries, PrimariesCIExy* PIK_RESTRICT xy) {
  switch (primaries) {
    case Primaries::kSRGB:
      xy->r.x = 0.639998686;
      xy->r.y = 0.330010138;
      xy->g.x = 0.300003784;
      xy->g.y = 0.600003357;
      xy->b.x = 0.150002046;
      xy->b.y = 0.059997204;
      return true;

    case Primaries::k2020:
      xy->r.x = 0.708;
      xy->r.y = 0.292;
      xy->g.x = 0.170;
      xy->g.y = 0.797;
      xy->b.x = 0.131;
      xy->b.y = 0.046;
      return true;

    case Primaries::kP3:
      xy->r.x = 0.680;
      xy->r.y = 0.320;
      xy->g.x = 0.265;
      xy->g.y = 0.690;
      xy->b.x = 0.150;
      xy->b.y = 0.060;
      return true;

    case Primaries::kAP0:
      xy->r.x = 0.7347;
      xy->r.y = 0.2653;
      xy->g.x = 0.0000;
      xy->g.y = 1.0000;
      xy->b.x = 0.0001;
      xy->b.y = -0.077;
      return true;

    case Primaries::kAP1:
      xy->r.x = 0.713;
      xy->r.y = 0.293;
      xy->g.x = 0.165;
      xy->g.y = 0.830;
      xy->b.x = 0.128;
      xy->b.y = 0.044;
      return true;

    case Primaries::kAdobe:
      xy->r.x = 0.639996511;
      xy->r.y = 0.329996864;
      xy->g.x = 0.210005295;
      xy->g.y = 0.710004866;
      xy->b.x = 0.149997606;
      xy->b.y = 0.060003644;
      return true;

    case Primaries::kUnknown:
      break;  // handled below
  }
  memset(xy, 0, sizeof(*xy));
  return false;
}

Primaries PrimariesFromCIExy(const PrimariesCIExy& xy) {
  if (ApproxEq(xy.r.x, 0.64) && ApproxEq(xy.r.y, 0.33) &&
      ApproxEq(xy.g.x, 0.30) && ApproxEq(xy.g.y, 0.60) &&
      ApproxEq(xy.b.x, 0.15) && ApproxEq(xy.b.y, 0.06)) {
    return Primaries::kSRGB;
  }

  if (ApproxEq(xy.r.x, 0.708) && ApproxEq(xy.r.y, 0.292) &&
      ApproxEq(xy.g.x, 0.170) && ApproxEq(xy.g.y, 0.797) &&
      ApproxEq(xy.b.x, 0.131) && ApproxEq(xy.b.y, 0.046)) {
    return Primaries::k2020;
  }
  if (ApproxEq(xy.r.x, 0.680) && ApproxEq(xy.r.y, 0.320) &&
      ApproxEq(xy.g.x, 0.265) && ApproxEq(xy.g.y, 0.690) &&
      ApproxEq(xy.b.x, 0.150) && ApproxEq(xy.b.y, 0.060)) {
    return Primaries::kP3;
  }
  if (ApproxEq(xy.r.x, 0.7347) && ApproxEq(xy.r.y, 0.2653) &&
      ApproxEq(xy.g.x, 0.0000) && ApproxEq(xy.g.y, 1.0000) &&
      ApproxEq(xy.b.x, 0.0001) && ApproxEq(xy.b.y, -0.077)) {
    return Primaries::kAP0;
  }
  if (ApproxEq(xy.r.x, 0.713) && ApproxEq(xy.r.y, 0.293) &&
      ApproxEq(xy.g.x, 0.165) && ApproxEq(xy.g.y, 0.830) &&
      ApproxEq(xy.b.x, 0.128) && ApproxEq(xy.b.y, 0.044)) {
    return Primaries::kAP1;
  }
  if (ApproxEq(xy.r.x, 0.64) && ApproxEq(xy.r.y, 0.33) &&
      ApproxEq(xy.g.x, 0.21) && ApproxEq(xy.g.y, 0.71) &&
      ApproxEq(xy.b.x, 0.15) && ApproxEq(xy.b.y, 0.06)) {
    return Primaries::kAdobe;
  }

  return Primaries::kUnknown;
}

double GammaFromTransferFunction(TransferFunction tf) {
  if (tf == TransferFunction::kLinear) return GammaLinear();
  if (tf == TransferFunction::kSRGB) return GammaSRGB();
  if (tf == TransferFunction::kAdobe) return GammaAdobe();
  if (tf == TransferFunction::k709) return Gamma709();
  if (tf == TransferFunction::kPQ) return GammaPQ();
  if (tf == TransferFunction::kHLG) return GammaHLG();
  return GammaUnknown();
}

TransferFunction TransferFunctionFromGamma(double gamma) {
  if (ApproxEq(gamma, GammaLinear())) return TransferFunction::kLinear;
  if (ApproxEq(gamma, GammaSRGB())) return TransferFunction::kSRGB;
  if (ApproxEq(gamma, GammaAdobe())) return TransferFunction::kAdobe;
  if (ApproxEq(gamma, Gamma709())) return TransferFunction::k709;
  if (ApproxEq(gamma, GammaPQ())) return TransferFunction::kPQ;
  if (ApproxEq(gamma, GammaHLG())) return TransferFunction::kHLG;
  if (ApproxEq(gamma, GammaUnknown())) return TransferFunction::kUnknown;
  return TransferFunction::kUnknown;
}

std::string StringFromWhitePoint(const CIExy& xy) {
  const WhitePoint wp = WhitePointFromCIExy(xy);
  if (wp != WhitePoint::kUnknown) return ToString(wp);
  std::string ret("WhitePoint:");
  ret += std::to_string(xy.x) + ",";
  ret += std::to_string(xy.y);
  return ret;
}

std::string StringFromPrimaries(const PrimariesCIExy& xy) {
  const Primaries primaries = PrimariesFromCIExy(xy);
  if (primaries != Primaries::kUnknown) return ToString(primaries);
  std::string ret("Primaries:");
  ret += std::to_string(xy.r.x) + ",";
  ret += std::to_string(xy.r.y) + ";";
  ret += std::to_string(xy.g.x) + ",";
  ret += std::to_string(xy.g.y) + ";";
  ret += std::to_string(xy.b.x) + ",";
  ret += std::to_string(xy.b.y);
  return ret;
}

std::string StringFromGamma(double gamma) {
  const TransferFunction tf = TransferFunctionFromGamma(gamma);
  if (tf != TransferFunction::kUnknown) return ToString(tf);
  return std::string("g") + std::to_string(gamma);
}

double GammaFromString(const std::string& s) {
  if (s.length() < 2) return 0.0;
  if (s[0] == 'g') {
    return stod(s.substr(1));
  }
  const auto transfer_function = ValueFromString<TransferFunction>(s);
  return GammaFromTransferFunction(transfer_function);
}

std::string Description(const ColorEncoding& c) {
  std::string description = ToString(c.color_space);

  if (c.color_space != ColorSpace::kXYZ) {
    description += "_" + ToString(c.white_point);
  }

  if (c.color_space != ColorSpace::kGray && c.color_space != ColorSpace::kXYZ) {
    description += "_" + ToString(c.primaries);
  }

  description += "_" + ToString(c.rendering_intent);

  description +=
      "_" + StringFromGamma(GammaFromTransferFunction(c.transfer_function));

  return description;
}

std::string Description(const ProfileParams& pp) {
  std::string description = ToString(pp.color_space);

  if (pp.color_space != ColorSpace::kXYZ) {
    description += "_" + ToString(WhitePointFromCIExy(pp.white_point));
  }

  if (pp.color_space != ColorSpace::kGray &&
      pp.color_space != ColorSpace::kXYZ) {
    description += "_" + ToString(PrimariesFromCIExy(pp.primaries));
  }

  description += "_" + ToString(pp.rendering_intent);

  // Gamma goes last for easier parsing.
  description += "_" + StringFromGamma(pp.gamma);

  return description;
}

Status ParseDescription(const std::string& description,
                        ProfileParams* PIK_RESTRICT pp) {
  // "Token" is a 3-character string followed by "_".
  class Tokenizer {
   public:
    Tokenizer(const std::string* tokens) : tokens_(tokens) {}
    Status Next(std::string* PIK_RESTRICT next) {
      if (pos_ + 4 > tokens_->length()) return PIK_FAILURE("String too short");
      if ((*tokens_)[pos_ + 3] != '_') return PIK_FAILURE("Missing terminator");
      *next = tokens_->substr(pos_, 3);
      pos_ += 4;
      return true;
    }

    std::string Tail() const { return tokens_->substr(pos_); }

   private:
    const std::string* tokens_;  // not owned
    size_t pos_ = 0;
  } tokenizer(&description);

  std::string next;
  PIK_RETURN_IF_ERROR(tokenizer.Next(&next));
  pp->color_space = ValueFromString<ColorSpace>(next);

  if (pp->color_space != ColorSpace::kXYZ) {
    PIK_RETURN_IF_ERROR(tokenizer.Next(&next));
    const WhitePoint white_point = ValueFromString<WhitePoint>(next);
    (void)WhitePointToCIExy(white_point, &pp->white_point);
  } else {
    memset(&pp->white_point, 0, sizeof(pp->white_point));
  }

  if (pp->color_space != ColorSpace::kGray &&
      pp->color_space != ColorSpace::kXYZ) {
    PIK_RETURN_IF_ERROR(tokenizer.Next(&next));
    const Primaries primaries = ValueFromString<Primaries>(next);
    (void)PrimariesToCIExy(primaries, &pp->primaries);
  } else {
    memset(&pp->primaries, 0, sizeof(pp->primaries));
  }

  PIK_RETURN_IF_ERROR(tokenizer.Next(&next));
  pp->rendering_intent = ValueFromString<RenderingIntent>(next);

  pp->gamma = GammaFromString(tokenizer.Tail());

  return true;
}

Status ColorEncodingToParams(const ColorEncoding& c,
                             ProfileParams* PIK_RESTRICT pp) {
  pp->color_space = c.color_space;
  pp->gamma = GammaFromTransferFunction(c.transfer_function);
  pp->rendering_intent = c.rendering_intent;

  // Avoid unnecessary failure by skipping white point/primaries if they are
  // undefined anyway.
  if (c.color_space != ColorSpace::kXYZ) {
    PIK_RETURN_IF_ERROR(WhitePointToCIExy(c.white_point, &pp->white_point));
  }

  if (c.color_space != ColorSpace::kGray && c.color_space != ColorSpace::kXYZ) {
    PIK_RETURN_IF_ERROR(PrimariesToCIExy(c.primaries, &pp->primaries));
  }

  return true;
}

void SetFieldsFromParams(const ProfileParams& pp,
                         ColorEncoding* PIK_RESTRICT c) {
  c->color_space = pp.color_space;
  c->white_point = WhitePointFromCIExy(pp.white_point);
  c->primaries = PrimariesFromCIExy(pp.primaries);
  c->transfer_function = TransferFunctionFromGamma(pp.gamma);
  c->rendering_intent = pp.rendering_intent;
}

std::vector<ColorEncoding> AllEncodings() {
  std::vector<ColorEncoding> all_encodings;
  all_encodings.reserve(300);
  ColorEncoding c;

  for (ColorSpace cs : AllValues<ColorSpace>()) {
    // TODO(janwas): support generating these
    if (cs == ColorSpace::kYCbCr || cs == ColorSpace::kICtCp) continue;
    c.color_space = cs;

    for (WhitePoint wp : AllValues<WhitePoint>()) {
      c.white_point = wp;
      // XYZ doesn't store a white point and retrieves E.
      if (cs == ColorSpace::kXYZ && wp != WhitePoint::kE) continue;

      for (Primaries primaries : AllValues<Primaries>()) {
        c.primaries = primaries;

        for (TransferFunction tf : AllValues<TransferFunction>()) {
          if (cs == ColorSpace::kXYZ && tf != TransferFunction::kLinear) {
            continue;
          }
          c.transfer_function = tf;

          for (RenderingIntent ri : AllValues<RenderingIntent>()) {
            c.rendering_intent = ri;

            all_encodings.push_back(c);
          }
        }
      }
    }
  }

  return all_encodings;
}

ColorEncoding::ColorEncoding() { Bundle::Init(this); }

}  // namespace pik
