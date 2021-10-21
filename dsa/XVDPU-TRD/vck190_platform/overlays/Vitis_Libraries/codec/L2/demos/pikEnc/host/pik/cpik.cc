// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/cpik.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/arch_specific.h"
#include "pik/args.h"
#include "pik/codec.h"
#include "pik/common.h"
#include "pik/file_io.h"
#include "pik/image.h"
#include "pik/os_specific.h"
#include "pik/padded_bytes.h"
#include "pik/pik.h"
#include "pik/pik_info.h"
#include "pik/profiler.h"
#include "pik/simd/targets.h"

namespace pik {
namespace {

// Proposes a distance to try for a given bpp target. This could depend
// on the entropy in the image, too, but let's start with something.
static double ApproximateDistanceForBPP(double bpp) {
  return 1.704 * pow(bpp, -0.804);
}

} // namespace

CompressArgs::CompressArgs() {
  // TODO(janwas): differentiate between cores/HT
  num_threads = AvailableCPUs().size() / 2;
}

Status CompressArgs::AddCommandLineOptions(tools::CommandLineParser *cmdline) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", "the input can be PNG PPM or PFM.",
                               &params.file_in);
  cmdline->AddPositionalOption(
      "OUTPUT", "the compressed output file (optional)", &params.file_out);

  // Flags.
  cmdline->AddOptionFlag('\0', "fast", "Use fast encoding mode (less dense).",
                         &params.fast_mode, &SetBooleanTrue);
  cmdline->AddOptionFlag('\0', "guetzli", "Use the guetzli mode.",
                         &params.guetzli_mode, &SetBooleanTrue);
  cmdline->AddOptionFlag('\0', "progressive", "Use the progressive mode.",
                         &params.progressive_mode, &SetBooleanTrue);
  cmdline->AddOptionFlag('\0', "lossless", "Use the lossless mode.",
                         &params.lossless_mode, &SetBooleanTrue);

  cmdline->AddOptionFlag('\0', "keep_tempfiles",
                         "Don't delete temporary files.",
                         &params.keep_tempfiles, &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "num_threads", "N",
                          "number of worker threads (zero = none).",
                          &num_threads, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "noise", "0|1",
                          "force enable/disable noise generation.",
                          &params.noise, &ParseOverride);

  cmdline->AddOptionValue('\0', "gradient", "0|1",
                          "force enable/disable extra gradient map.",
                          &params.gradient, &ParseOverride);
  cmdline->AddOptionValue('\0', "adaptive_reconstruction", "0|1",
                          "force enable/disable decoder filter.",
                          &params.adaptive_reconstruction, &ParseOverride);

  cmdline->AddOptionValue('\0', "gaborish", "0..7",
                          "chooses deblocking strength (4=normal).",
                          &params.gaborish, &ParseGaborishStrength);

  cmdline->AddOptionValue('\0', "xclbin", "string",
                          "path to xclbin file",
                          &params.xclbinPath, &ParseString);

  // Target distance/size/bpp
  opt_distance_id = cmdline->AddOptionValue(
      '\0', "distance", "maxError",
      ("Max. butteraugli distance, lower = higher quality.\n"
       "    Good default: 1.0. Supported range: 0.5 .. 3.0."),
      &params.butteraugli_distance, &ParseFloat);
  opt_target_size_id = cmdline->AddOptionValue(
      '\0', "target_size", "N",
      ("Aim at file size of N bytes.\n"
       "    Compresses to 1 % of the target size in ideal conditions.\n"
       "    Runs the same algorithm as --target_bpp"),
      &params.target_size, &ParseUnsigned);
  opt_target_bpp_id = cmdline->AddOptionValue(
      '\0', "target_bpp", "BPP",
      ("Aim at file size that has N bits per pixel.\n"
       "    Compresses to 1 % of the target BPP in ideal conditions."),
      &params.target_bitrate, &ParseFloat);

  cmdline->AddOptionValue(
      '\0', "intensity_target", "N",
      ("Intensity target of monitor in nits, higher\n"
       "   results in higher quality image. Supported range: 250..6000,\n"
       "   default is 250."),
      &params.intensity_target, &ParseFloat);

  cmdline->AddOptionValue('\0', "saliency_extractor", "STRING", nullptr,
                          &params.saliency_extractor_for_progressive_mode,
                          &ParseString);
  cmdline->AddOptionValue('\0', "saliency_threshold", "N", nullptr,
                          &params.saliency_threshold, &ParseFloat);
  cmdline->AddOptionFlag('\0', "saliency_debug_skip_nonsalient", nullptr,
                         &params.saliency_debug_skip_nonsalient,
                         &SetBooleanTrue);

  cmdline->AddOptionValue(
      'x', "dec-hints", "key=value",
      "color_space indicates the ColorEncoding, see Description().", &dec_hints,
      &ParseAndAppendKeyValue);

  cmdline->AddOptionFlag('v', "verbose",
                         "enable verbose mode with additional output",
                         &params.verbose, &SetBooleanTrue);
  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);
  return true;
}

Status CompressArgs::ValidateArgs(const tools::CommandLineParser &cmdline) {
  bool got_distance = cmdline.GetOption(opt_distance_id)->matched();
  bool got_target_size = cmdline.GetOption(opt_target_size_id)->matched();
  bool got_target_bpp = cmdline.GetOption(opt_target_bpp_id)->matched();

  if (got_target_size) {
    fprintf(stderr, "Warning: target_size does not set all flags/modes.\n");
  }
  if (got_target_bpp) {
    fprintf(stderr, "Warning: target_bpp does not set all flags/modes.\n");
  }
  if (got_distance) {
    constexpr float butteraugli_min_dist = 0.125f;
    constexpr float butteraugli_max_dist = 15.0f;
    if (!(butteraugli_min_dist <= params.butteraugli_distance &&
          params.butteraugli_distance <= butteraugli_max_dist)) {
      fprintf(stderr, "Invalid/out of range distance, try %g to %g.\n",
              butteraugli_min_dist, butteraugli_max_dist);
      return false;
    }
  }

  if (got_target_bpp + got_target_size + got_distance > 1) {
    fprintf(stderr,
            "You can specify only one of '--distance', "
            "'--target_bpp' and '--target_size'. They are all different ways"
            " to specify the image quality. When in doubt, use --distance."
            " It gives the most visually consistent results.\n");
    return false;
  }

  if (!params.saliency_extractor_for_progressive_mode.empty()) {
    if (!params.progressive_mode) {
      fprintf(stderr,
              "Warning: Specifying --saliency_extractor only makes sense "
              "for --progressive mode.\n");
    }
    if (!params.file_out) {
      fprintf(stderr,
              "Need to have output filename to use saliency extractor.\n");
      return PIK_FAILURE("file_out");
    }
  }

  if (!params.file_in) {
    fprintf(stderr, "Missing input filename.\n");
    return false;
  }

  return true;
}

Status Compress(ThreadPool *pool, std::string xclbinPath, CompressArgs &args,
                PaddedBytes *compressed) {
  double t0, t1;

  CodecContext codec_context;
  CodecInOut io(&codec_context);
  io.dec_hints = args.dec_hints;
  t0 = Now();
  if (!io.SetFromFile(args.params.file_in)) {
    fprintf(stderr, "Failed to read image %s.\n", args.params.file_in);
    return false;
  }
  t1 = Now();
  const double decode_mps = io.xsize() * io.ysize() * 1E-6 / (t1 - t0);

  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  if (args.params.target_size > 0 || args.params.target_bitrate > 0) {
    // Search algorithm for target bpp / size.
    CompressArgs s = args; // Args for search.
    if (s.params.target_size > 0) {
      s.params.target_bitrate =
          s.params.target_size * 8.0 / (io.xsize() * io.ysize());
      s.params.target_size = 0;
    }
    double dist = ApproximateDistanceForBPP(s.params.target_bitrate);
    s.params.butteraugli_distance = dist;
    double target_size =
        s.params.target_bitrate * (1 / 8.) * io.xsize() * io.ysize();
    s.params.target_bitrate = 0;
    double best_dist = 1.0;
    double best_loss = 1e99;
    for (int i = 0; i < 7; ++i) {
      s.params.butteraugli_distance = dist;
      PaddedBytes candidate;
      bool ok = Compress(pool, xclbinPath, s, &candidate);
      if (!ok) {
        printf("Compression error occurred during the search for best size."
               " Trying with butteraugli distance %.15g\n",
               best_dist);
        break;
      }
      printf("Butteraugli distance %g yields %zu bytes, %g bpp.\n", dist,
             candidate.size(),
             candidate.size() * 8.0 / (io.xsize() * io.ysize()));
      const double ratio = static_cast<double>(candidate.size()) / target_size;
      const double loss = std::max(ratio, 1.0 / std::max(ratio, 1e-30));
      if (best_loss > loss) {
        best_dist = dist;
        best_loss = loss;
      }
      dist *= ratio;
      if (dist < 0.01) {
        dist = 0.01;
      }
      if (dist >= 16.0) {
        dist = 16.0;
      }
    }
    printf("Choosing butteraugli distance %.15g\n", best_dist);
    args.params.butteraugli_distance = best_dist;
    args.params.target_bitrate = 0;
    args.params.target_size = 0;
  }
  char mode[200];
  if (args.params.fast_mode) {
    strcpy(mode, "in fast mode ");
  }
  snprintf(mode, sizeof(mode), "with maximum Butteraugli distance %f",
           args.params.butteraugli_distance);
  fprintf(stderr,
          "Read %zu bytes (%zux%zu, %.1f MP/s); compressing %s, %zu threads.\n",
          io.enc_size, xsize, ysize, decode_mps, mode, NumWorkerThreads(pool));

  PikInfo aux_out;
  if (args.inspector_image3f) {
    aux_out.SetInspectorImage3F(args.inspector_image3f);
  }
  t0 = Now();
  if (!PixelsToPik(args.params, xclbinPath, &io, compressed, &aux_out, pool)) {
    fprintf(stderr, "Failed to compress.\n");
    return false;
  }
  t1 = Now();
  const size_t channels = io.c_current().Channels() + io.HasAlpha();
  const size_t bytes = xsize * ysize * channels *
                       DivCeil(io.original_bits_per_sample(), kBitsPerByte);
  const double bpp =
      static_cast<double>(compressed->size() * kBitsPerByte) / (xsize * ysize);
  fprintf(stderr, "Compressed to %zu bytes (%.3f bpp, %.2f MB/s).\n",
          compressed->size(), bpp, bytes * 1E-6 / (t1 - t0));

  if (args.params.verbose) {
    aux_out.Print(1);
  }

  return true;
}

} // namespace pik
